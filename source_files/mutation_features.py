import pandas as pd
import numpy as np
import sys
import os
import pickle
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix
import time

class mutationFeatures:
    """
    A class for creating the feature matrix of mutations for a set of samples and CpG sites 
    whose methylation we want to predict
    """
    def __init__(
        self, 
        all_mut_w_age_df: pd.DataFrame,
        illumina_cpg_locs_df: pd.DataFrame, 
        all_methyl_age_df_t: pd.DataFrame,
        out_dir: str,
        consortium: str,
        dataset: str,
        cross_val_num: int,
        matrix_qtl_dir: str
        #meqtl_db: pd.DataFrame = None
        ):
        self.all_mut_w_age_df = all_mut_w_age_df
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        #self.meqtl_db = meqtl_db
        self.out_dir = out_dir
        self.dataset = dataset
        self.consortium = consortium
        self.cross_val_num = cross_val_num
        # pre-process the mutation and methylation data
        self._preproc_mut_and_methyl()
        # choose train and test samples based on cross validation number
        self.train_samples, self.test_samples = self.cross_val_samples()
        self.matrix_qtl_dir = matrix_qtl_dir
        # create empty cache
        self.matrixQTL_store = {}
        self.all_samples = self.all_methyl_age_df_t.index.to_list()
        # create empty feature store
        self.mutation_features_store = {}
    
    def _preproc_mut_and_methyl(
        self
        ) -> None:
        """
        Create all_mut_w_age_illum_df, remove X & Y chromosomes,
        and only keep samples with measured methylation
        """
        # if a dataset is specified, subset the data to only this tissue type
        if self.dataset != "":
            if self.dataset == 'RCC':
                RCC_datasets = ['KIRC', 'KIRP' , 'KICH']
                self.all_methyl_age_df_t = self.all_methyl_age_df_t.loc[
                    self.all_methyl_age_df_t['dataset'].isin(RCC_datasets), :]
                self.all_methyl_age_df_t['dataset'] = 'RCC'
                self.all_mut_w_age_df = self.all_mut_w_age_df.loc[
                    self.all_mut_w_age_df['dataset'].isin(RCC_datasets), :].copy(deep=True)
                self.all_mut_w_age_df['dataset'] = 'RCC'
            else:
                self.all_methyl_age_df_t = self.all_methyl_age_df_t.loc[
                    self.all_methyl_age_df_t['dataset'] == self.dataset, :]
                self.all_mut_w_age_df = self.all_mut_w_age_df.loc[
                    self.all_mut_w_age_df['dataset'] == self.dataset, :].copy(deep=True)
            
        # if a mut_loc column does not exit, add it
        if 'mut_loc' not in self.all_mut_w_age_df.columns:
            self.all_mut_w_age_df['mut_loc'] = self.all_mut_w_age_df['chr'] + ':' \
                                                + self.all_mut_w_age_df['start'].astype(str)
        # mutations: non X and Y chromosomes and occured in samples with measured methylation
        self.all_mut_w_age_df = self.all_mut_w_age_df.loc[
            (self.all_mut_w_age_df['chr'] != 'X') 
            & (self.all_mut_w_age_df['chr'] != 'Y')
            & (self.all_mut_w_age_df['chr'] != 'MT')
            & (self.all_mut_w_age_df['case_submitter_id'].isin(self.all_methyl_age_df_t.index)),
            :]
        # join self.all_mut_w_age_df with the illumina_cpg_locs_df
        all_mut_w_age_illum_df = self.all_mut_w_age_df.copy(deep=True)
        all_mut_w_age_illum_df['start'] = pd.to_numeric(self.all_mut_w_age_df['start'])
        self.all_mut_w_age_illum_df = all_mut_w_age_illum_df.merge(
                                        self.illumina_cpg_locs_df, on=['chr', 'start'], how='left'
                                        )
        # subset illumina_cpg_locs_df to only the CpGs that are measured, and remove XY
        self.illumina_cpg_locs_df = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)
            & (self.illumina_cpg_locs_df['chr'] != 'X') 
            & (self.illumina_cpg_locs_df['chr'] != 'Y')
            ]
        # drop CpGs that are not in the illumina_cpg_locs_df (i.e. on XY)
        self.all_methyl_age_df_t = self.all_methyl_age_df_t.loc[:, 
            list(set(self.all_methyl_age_df_t.columns).intersection(
                set(self.illumina_cpg_locs_df['#id'].to_list() + ['dataset', 'gender', 'age_at_index'])
                ))
            ]
        # one hot encode covariates
        if self.dataset == "":
            dset_col = self.all_methyl_age_df_t['dataset'].to_list()
            self.all_methyl_age_df_t = pd.get_dummies(self.all_methyl_age_df_t, columns=["gender", "dataset"])
            # add back in the dataset column
            self.all_methyl_age_df_t['dataset'] = dset_col
        else: # only do gender if one dataset is specified
            self.all_methyl_age_df_t = pd.get_dummies(self.all_methyl_age_df_t, columns=["gender"])
        # subset meqtl_db to only cpgs in all_methyl_age_df_t 
        #self.meqtl_db = self.meqtl_db.loc[self.meqtl_db['cpg'].isin(self.all_methyl_age_df_t.columns), :]
    
    def cross_val_samples(self):
        """
        Choose train and test samples based on cross validation number and dataset
        @ return: train_samples, test_samples
        """
        # implicitly subsets to only this dataset's samples bc of preproc_mut_and_methyl
        skf = StratifiedKFold(n_splits=3, random_state=10, shuffle=True)
        # select the self.cross_val_num fold
        for i, (train_index, test_index) in enumerate(skf.split(self.all_methyl_age_df_t, self.all_methyl_age_df_t.loc[:, 'age_at_index'])):
            if i == self.cross_val_num:
                train_samples = self.all_methyl_age_df_t.iloc[train_index].index.to_list()
                test_samples = self.all_methyl_age_df_t.iloc[test_index].index.to_list()
                break
        return train_samples, test_samples

    def _select_correl_sites(
        self,
        cpg_id: str,
        cpg_chr: str,
        num_correl_sites: int,
        ) -> dict:
        """
        Just in time correlation to find the most correlated sites to the mutation event CpG in matched train_samples
        @ cpg_id: the id cpg to corr with
        @ cpg_chr: the chromosome of the cpg
        @ num_correl_sites: the number of sites to return, half pos half neg
        @ return: dict of {pos_cor: list of pos correlated positions, neg_cor: list of neg correlated positions}
        """
        # get cpg_id's MF
        cpg_mf = self.all_methyl_age_df_t.loc[self.train_samples, cpg_id]
        # get the MF of all same chrom CpGs
        same_chrom_cpgs = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['chr'] == cpg_chr, '#id'].values
        same_chrom_cpgs_mf = self.all_methyl_age_df_t.loc[self.train_samples, same_chrom_cpgs]
        # get correlation between mut_cpg and all same chrom CpGs
        corrs = same_chrom_cpgs_mf.corrwith(cpg_mf, axis=0)
        corrs.sort_values(ascending=True, inplace=True)
        idx = corrs.index.to_list()
        pos_corrs = idx[-int(num_correl_sites/2):]
        neg_corrs = idx[:int(num_correl_sites/2)]
        # convert cpg id's to locations
        pos_corr_locs = (
            self.illumina_cpg_locs_df.loc[
                self.illumina_cpg_locs_df['#id'].isin(pos_corrs)
                ].assign(location=lambda df: df['chr'] + ':' + df['start'].astype(str))['location']
            .tolist()
            )
        neg_corr_locs = (
            self.illumina_cpg_locs_df.loc[
                self.illumina_cpg_locs_df['#id'].isin(neg_corrs)
                ].assign(location=lambda df: df['chr'] + ':' + df['start'].astype(str))['location']
            .tolist()
            )
        return pos_corr_locs, neg_corr_locs
    
    def _get_matrixQTL_sites(
        self,
        cpg_id: str,
        chrom: str,
        max_meqtl_sites: int
        ) -> dict:
        """
        Given a CpG, get the max_meqtl_sites number meQTLs with smallest p-value in relation to this CpG
        @ cpg_id: the CpG id
        @ chrom: the chromosome of the CpG, to read in the correct matrixQTL results
        @ max_meqtl_sites: the maximum number of meQTLs to return
        @ return: a dict of the meQTLs split by beta, 'chr:start', with smallest p-value
        """
        # if chrom is not in the keys of the matrixQTL_store
        if chrom not in self.matrixQTL_store:
            # read in the matrixQTL results for this chromosome        
            meqtl_df = pd.read_parquet(
                os.path.join(self.matrix_qtl_dir, f"chr{chrom}_meqtl_fold_{self.cross_val_num}.parquet"),
                columns=['#id', 'SNP', 'p-value', 'beta', 'distance', 'snp_chr'])       
            self.matrixQTL_store[chrom] = meqtl_df
        else:
            meqtl_df = self.matrixQTL_store[chrom]
        # same chromosome only
        cis_meqtl_df = meqtl_df.loc[(meqtl_df['snp_chr'] == chrom)
                                    & (meqtl_df['#id'] == cpg_id) 
                                    , :]
        # split by beta and distance
        pos_meqtls = cis_meqtl_df.loc[
            (cis_meqtl_df['beta'] > 0),
            :]
        close_pos_meqtls_l = pos_meqtls.loc[
            pos_meqtls['distance'] < 5e6, # within 5Mb
            :].nsmallest(max_meqtl_sites, 'p-value')['SNP'].to_list()
        pos_meqtls_l = pos_meqtls.nsmallest(max_meqtl_sites, 'p-value')['SNP'].to_list()
        # negative
        neg_meqtls = cis_meqtl_df.loc[
            (cis_meqtl_df['beta'] < 0),
            :]
        close_neg_meqtls_l = neg_meqtls.loc[
            neg_meqtls['distance'] < 5e6, 
            :].nsmallest(max_meqtl_sites, 'p-value')['SNP'].to_list()
        neg_meqtls_l = neg_meqtls.nsmallest(max_meqtl_sites, 'p-value')['SNP'].to_list()
        return neg_meqtls_l, close_neg_meqtls_l, pos_meqtls_l, close_pos_meqtls_l
    
    def _select_db_sites(self, cpg_id, num_db_sites):
        this_cpg_meqtls = self.meqtl_db[self.meqtl_db['cpg'] == cpg_id]
        neg_cpg_meqtls = this_cpg_meqtls[this_cpg_meqtls['beta'] < 0].nsmallest(num_db_sites, 'beta')['snp'].to_list()
        pos_cpg_meqtls = this_cpg_meqtls[this_cpg_meqtls['beta'] > 0].nlargest(num_db_sites, 'beta')['snp'].to_list()
        return neg_cpg_meqtls, pos_cpg_meqtls
    
    def _get_predictor_site_groups(
        self, 
        cpg_id: str,
        num_correl_sites: int, # get extended
        max_meqtl_sites: int, # get extended
        nearby_window_size: int, 
        extend_amount: int
        ) -> list:
        """
        Get the sites to be used as predictors of cpg_id's methylation
        @ cpg_id: the id of the CpG
        @ num_correl_sites: the number of correlated sites to be used
        @ max_meqtl_sites: the maximum number of meQTLs to be used
        @ nearby_window_size: the window size to be used to find nearby sites
        @ returns: dict of types of genomic locations of the sites to be used as predictors in format chr:start
        """
        def extend(loc_list, extend_amount):
            """
            Return the list of locations with each original loc extended out extend_amount in each direction
            """
            return [loc.split(':')[0] + ':' + str(int(loc.split(':')[1]) + i) 
                    for loc in loc_list 
                    for i in range(-extend_amount, extend_amount + 1)]
            
        def extend_clump_matrixQTL_sites(loc_list, extend_amount = 1000):
            """For these extend 1000bp out to right, because this is the clumping window"""
            return [loc.split(':')[0] + ':' + str(int(loc.split(':')[1]) + i) 
                    for loc in loc_list
                    for i in range(extend_amount + 1)
                    ]
        
        predictor_site_groups = {}
        try: # get cpg_id's chromosome and start position
            chrom = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'] == cpg_id, 'chr'].values[0]
            start = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'] == cpg_id, 'start'].values[0]
        except: # for some reason this cpg is not in illumina_cpg_locs_df, return empty dict
            return {}
        # get dict of positions of most positively and negatively correlated CpGs
        predictor_site_groups['pos_corr'], predictor_site_groups['neg_corr'] = self._select_correl_sites(
            cpg_id, chrom, num_correl_sites=num_correl_sites
            )
        # get extended versions of the num_correl_ext_sites most positively and negatively correlated CpGs
        predictor_site_groups['pos_corr_ext'] = extend(
            predictor_site_groups['pos_corr'], extend_amount=extend_amount
            )
        predictor_site_groups['neg_corr_ext'] = extend(
            predictor_site_groups['neg_corr'], extend_amount=extend_amount
            )
        # get nearby sites
        predictor_site_groups['nearby'] = [
            chrom + ':' + str(start + i)
            for i in range(-int(nearby_window_size/2), int(nearby_window_size/2) + 1)
            ]
        predictor_site_groups['very_nearby'] = [
            chrom + ':' + str(start + i)
            for i in range(-int(nearby_window_size/20), int(nearby_window_size/20) + 1)
            ]
        # get sites from matrixQTL 
        predictor_site_groups['matrixqtl_neg_beta'], predictor_site_groups['matrixqtl_neg_beta_close'],  predictor_site_groups['matrixqtl_pos_beta'],  predictor_site_groups['matrixqtl_pos_beta_close'] = self._get_matrixQTL_sites(cpg_id, chrom, max_meqtl_sites=max_meqtl_sites)
        # extend matrixQTL sites
        predictor_site_groups['matrixqtl_neg_beta_ext'] = extend_clump_matrixQTL_sites(
            predictor_site_groups['matrixqtl_neg_beta'], extend_amount=1000
            )
        predictor_site_groups['matrixqtl_neg_beta_close_ext'] = extend_clump_matrixQTL_sites(
            predictor_site_groups['matrixqtl_neg_beta_close'], extend_amount=1000
            )
        predictor_site_groups['matrixqtl_pos_beta_ext'] = extend_clump_matrixQTL_sites(
            predictor_site_groups['matrixqtl_pos_beta'], extend_amount=1000
        )
        predictor_site_groups['matrixqtl_pos_beta_close_ext'] = extend_clump_matrixQTL_sites(
            predictor_site_groups['matrixqtl_pos_beta_close'], extend_amount=1000
        )
        # get database meQtls
        """predictor_site_groups['db_neg_beta'], predictor_site_groups['db_pos_beta'] = self._select_db_sites(cpg_id, num_db_sites)
        predictor_site_groups['db_neg_beta_ext'] = extend(
            predictor_site_groups['db_neg_beta'], extend_amount=extend_amount
            )
        predictor_site_groups['db_pos_beta_ext'] = extend(
            predictor_site_groups['db_pos_beta'], extend_amount=extend_amount
            )"""
        return predictor_site_groups
    
    def _create_feature_mat(
        self, 
        cpg_id: str, 
        predictor_groups: dict,
        aggregate: str,
        extend_amount: int,
        binarize: bool,
        ) -> tuple:
        """
        Create the training matrix for the given cpg_id and predictor_sites
        @ cpg_id: the id of the CpG
        @ predictor_groups: dict of lists of sites to be used as predictors of cpg_id's methylation
        @ aggregate: whether to aggregate the mutation status by predictor group
        @ extend_amount: the amount the predictor sites were extended by
        @ returns:
            - feature_mat: a feature matrix where each column is a predictor site from predictor_groups, sparse matrix
            - predictor_sites: the list of predictor sites in the feature matrix (columns)
            - target_values: the target values for the feature matrix (methylation values)
        """
        # make list of all samples to fill in samples that do not have mutations in predictor sites
        all_samples = self.train_samples + self.test_samples
        
        def _create_feat_mat(
            site_list: list, 
            binarize: bool = False
            ) -> pd.DataFrame:
            mut_status = self.all_mut_w_age_df.loc[
                self.all_mut_w_age_df['mut_loc'].isin(site_list),
                ['DNA_VAF', 'case_submitter_id', 'mut_loc']
                ]
            feat_mat = pd.pivot_table(
                mut_status, index='case_submitter_id', columns='mut_loc',
                values='DNA_VAF', fill_value = 0
                )
            feat_mat = feat_mat.reindex(all_samples, fill_value=0)
            if binarize:
                # convert all nonzero values to 1
                feat_mat[feat_mat > 0] = 1
            return feat_mat
        
        def noAgg() -> pd.DataFrame:
            """
            Create a feature matrix where each column is a predictor site from predictor_groups
            and values are the variant allele frequencies of the mutations at that site in the sample
            """
            # get list of unique predictor sites
            predictor_sites = set()
            for key in predictor_groups:
                predictor_sites.update(predictor_groups[key])
            predictor_sites = list(predictor_sites)
            # get mutation status of all predictor sites
            feat_mat = _create_feat_mat(predictor_sites, binarize=binarize)
            return feat_mat
        
        def agg(
            extended_feat_agg: bool = True
            ) -> pd.DataFrame:
            """
            Create a feature matrix where each column is a predictor group from predictor_groups
            and values are the sum of the variant allele frequencies of the mutations in
            that group in the sample
            @ extended_feat_agg: Turn each set of sites extended from one site into their own feature.
            So if chr1:500 was extended by 250, then chr1:250-750 will be a feature.
            """
            # list of dataframes containing aggregated features
            aggregated_muts = []
            extended_feat_agg_muts = []
            # for each predictor group, e.g. pos_corr, neg_corr, etc.
            for group_name, group_sites in predictor_groups.items():
                # create feature matrix for this group
                feat_mat = _create_feat_mat(group_sites, binarize=binarize)
                # sum across loci within each sample to get samples x aggregate feature matrix
                agg_feat_mat = feat_mat.sum(axis=1)
                aggregated_muts.append(agg_feat_mat)
                
                ### Aggregation across each extended site for each sample ###
                # if we want to add, the group name ends with '_ext', 
                # and there are more than 0 mutated sites in the group 
                if extended_feat_agg and group_name.endswith('_ext') and feat_mat.shape[1] > 0:
                    # sum every extend_amount columns of feat_mat
                    number_ext_features = (len(group_sites) // extend_amount)
                    sums, col_names = [], []
                    for i in range(number_ext_features - 1):
                        # sum the features in the ith extended feature that are in feat_mat
                        to_sum = list(set(group_sites[i*extend_amount:(i+1)*extend_amount]).intersection(feat_mat.columns))
                        sums.append(feat_mat.loc[:, to_sum].sum(axis=1))
                        # create the name of the ith extended feature
                        name = group_sites[i*extend_amount] \
                                            + '-' \
                                            + group_sites[(i+1)*extend_amount].split(':')[1] \
                                            + '-' + group_name
                        col_names.append(name)
                    extended_feats = pd.concat(sums, axis=1)
                    # rename features 
                    extended_feats.columns = col_names
                    # add extended features to extended_feat_agg_muts
                    extended_feat_agg_muts.append(extended_feats)
                
            # create dataframe from aggreated mutations with columns named after pred_type
            agg_feat_mat = pd.concat(aggregated_muts, axis=1)
            agg_feat_mat.columns = predictor_groups.keys()
            # create df from extended features
            extended_feat_agg_muts = pd.concat(extended_feat_agg_muts, axis=1)
            # add extended features to agg_feat_mat
            agg_feat_mat = pd.concat([agg_feat_mat, extended_feat_agg_muts], axis=1)
            return agg_feat_mat
        
        def get_tesselated_nearby_feats(
            tesselate_sizes = [15, 50, 100, 200, 500, 1000]
            ) -> pd.DataFrame:
            """
            For the 'nearby' predictor group, create a feature matrix with these features aggregated into
            5bp, 10bp, 25bp, 50bp, 200bp windows
            """
            def find_contiguous_sets(input_list, N):
                def subtract_special(x, y):
                    x = int(x.split(':')[1])
                    y = int(y.split(':')[1])
                    return x - y
                sets = []
                curr_set = []
                i = 0
                for i in range(len(input_list)):
                    if not curr_set:
                        curr_set.append(input_list[i])
                    elif subtract_special(input_list[i], curr_set[0]) < N:
                        curr_set.append(input_list[i])
                    else:
                        sets.append(curr_set)
                        curr_set = [input_list[i]]
                if curr_set:
                    sets.append(curr_set)
                # remove lists that are 1 element long
                sets = [s for s in sets if len(s) > 1]
                return sets
            
            def sum_columns(df, N):
                """
                Given a df, sum every N columns together
                """
                old_cols = df.columns.to_list()
                sets_to_sum = find_contiguous_sets(old_cols, N)
                new_df_dict = {}
                # for each list of sites
                for cols in sets_to_sum:
                    # select these sites
                    to_sum = df[cols]
                    # sum them, naming the new column
                    new_df_dict[f"{to_sum.columns[0]}-{to_sum.columns[-1]}_{N}_tesselated"] = to_sum.sum(axis=1)
                return pd.DataFrame(new_df_dict)
            
            nearby_sites = predictor_groups['nearby']
            # create feature matrix for nearby sites
            feat_mat = _create_feat_mat(nearby_sites, binarize=binarize)
            # sum features in tesselate_sizes bp windows
            all_tesse_feat_mats = []
            for tesse_size in tesselate_sizes:
                this_tesse_size_feat_mat = sum_columns(feat_mat, tesse_size) 
                all_tesse_feat_mats.append(this_tesse_size_feat_mat)               
            tesselated_feat_mats = pd.concat(all_tesse_feat_mats, axis=1)
            return tesselated_feat_mats
        
        def get_nested_nearby_feats(num_nested_feats: int = 200):
            """
            For nest_size / 2 in each direction from mutation aggregate nearby features
            """
            # get nearby sites and window size
            nearby_sites = predictor_groups['nearby']
            nearby_window_size = len(nearby_sites)
            # log increasing values from 10 to nearby window size
            nested_sizes = np.logspace(
                start = 1, stop = np.log10(nearby_window_size), num = num_nested_feats, base = 10
                ).astype(int)
            # mutationn statuis
            feat_mat = _create_feat_mat(nearby_sites, binarize=binarize)
            # find middle of window (where mutated site is)
            middle_index = int(len(nearby_sites) / 2)
            all_nested_feat_mats = []
            nested_col_names = []
            for nest_size in nested_sizes:
                # we know that the mutated site is in the middle of the window
                # so extend out nest_size / 2 in each direction from middle and aggregate
                # the features in that window
                start = middle_index - int(nest_size / 2)                
                end = middle_index + int(nest_size / 2) + 1
                # select these sites from nearby sites
                this_nest_nearby_sites = nearby_sites[start:end]
                # select these sites from feat mat, if they exist
                this_nest_nearby_sites = list(set(this_nest_nearby_sites).intersection(set(feat_mat.columns)))
                if len(this_nest_nearby_sites) == 0:
                    continue
                this_nested_size_feat_mat = feat_mat[this_nest_nearby_sites].sum(axis=1)
                # add to list of all nested feat mats and col names
                all_nested_feat_mats.append(this_nested_size_feat_mat)
                nested_col_names.append(
                    f"{this_nest_nearby_sites[0]}-{this_nest_nearby_sites[-1]}_{nest_size}_nested"
                    )
            if len(all_nested_feat_mats) == 0:
                return None
            else:
                nested_feat_mat = pd.concat(all_nested_feat_mats, axis=1)
                nested_feat_mat.columns = nested_col_names
                return nested_feat_mat
 
        if aggregate == "Both":
            feat_mat = noAgg()
            agg_feat_mat = agg()
            tesselated_nearby_feats = get_tesselated_nearby_feats()
            nested_nearby_feats = get_nested_nearby_feats()
            feat_mat = pd.merge(feat_mat, agg_feat_mat, left_index=True, right_index=True)
            feat_mat = pd.merge(feat_mat, tesselated_nearby_feats, left_index=True, right_index=True)
            if nested_nearby_feats is not None:
                feat_mat = pd.merge(feat_mat, nested_nearby_feats, left_index=True, right_index=True)
        elif aggregate == "False":
            feat_mat = noAgg()
        elif aggregate == "True":
            agg_feat_mat = agg()
            tesselated_nearby_feats = get_tesselated_nearby_feats()
            nested_nearby_feats = get_nested_nearby_feats()
            feat_mat = pd.merge(agg_feat_mat, tesselated_nearby_feats, left_index=True, right_index=True)
            if nested_nearby_feats is not None:
                feat_mat = pd.merge(feat_mat, nested_nearby_feats, left_index=True, right_index=True)
        else:
            sys.exit("Aggregate must be either 'True', 'False', or Both")
        # add covariate columns to X
        coviariate_col_names = [
            col for  col in self.all_methyl_age_df_t.columns \
            if col.startswith('dataset_') or col.startswith('gender_')
            ]
        covariate_df = self.all_methyl_age_df_t.loc[all_samples, coviariate_col_names]
        # right merge to make sure all samples are included
        feat_mat = pd.merge(
            feat_mat, covariate_df, 
            left_index=True, right_index=True, how = 'right'
            )
        # convert to float16 to save memory
        feat_mat = feat_mat.astype('float16')
        # drop duplicate columns
        try:
            feat_mat = feat_mat.loc[all_samples, ~feat_mat.columns.duplicated()]
        except: # if there is any error 
            print(feat_mat)
            print(set(all_samples) - set(feat_mat.index))
            print(len(set(all_samples) - set(feat_mat.index)))
            print(len(feat_mat.index))
        # convert feat_mat to sparse matrix
        feature_names = feat_mat.columns.to_list()
        feat_mat = csr_matrix(feat_mat)
        # get MF target values
        target_values = self.all_methyl_age_df_t.loc[all_samples, cpg_id]
        # within each dataset convert each value to a MD score within that dataset
        def madd(x):
            return (x - x.median()).abs().median()
        mad_target_values = self.all_methyl_age_df_t.loc[
            all_samples, [cpg_id, 'dataset']
            ].groupby('dataset').transform(lambda x: (x - x.median()).div(madd(x)))
        # make sure same order and a series
        mad_target_values = mad_target_values.loc[all_samples, cpg_id] 
        return feat_mat, feature_names, target_values, mad_target_values
    
    def create_all_feat_mats(
        self, 
        cpg_ids: list, 
        aggregate: str,
        num_correl_sites: int = 50,
        max_meqtl_sites: int = 100,
        nearby_window_size: int = 50000,
        extend_amount: int = 250,
        binarize: bool = False
        ):
        """
        Create the training matrix for the given cpg_id and predictor_sites
        @ cpg_ids: the ids of the CpGs
        @ predictor_groups: dict of lists of sites to be used as predictors of cpg_id's methylation
        @ samples: the samples to be included in the training matrix
        @ aggregate: whether to aggregate the mutation status by predictor group
        @ returns: None
        """
        feat_mats, feat_names, target_values, mad_target_values = {}, {}, {}, {}
        for i, cpg_id in enumerate(cpg_ids):
            # first get the predictor groups
            predictor_groups = self._get_predictor_site_groups(
                cpg_id, num_correl_sites, max_meqtl_sites,
                nearby_window_size, extend_amount
                )
            
            feat_mats[cpg_id], feat_names[cpg_id], target_values[cpg_id], mad_target_values[cpg_id] = self._create_feature_mat(
                cpg_id, predictor_groups, aggregate, extend_amount, binarize
                )
            if i % 10 == 0:
                print(f"Finished {i} of {len(cpg_ids)}", flush=True)
    
        if len(self.mutation_features_store) == 0:
            # create a dictionary to allow for easy data persistence
            # to get the feature matrix for a given cpg_id, use feat_mats[cpg_id]
            # the feature names are in feat_names[cpg_id]
            # and the sample order is in target_values[cpg_id].index
            self.mutation_features_store = {
                'dataset': self.dataset, # string
                'train_samples': self.train_samples, # list
                'test_samples': self.test_samples, # list
                'aggregate': aggregate, # string
                'num_correl_sites': num_correl_sites, # int
                'max_meqtl_sites': max_meqtl_sites, # int
                'nearby_window_size': nearby_window_size, # int
                'extend_amount': extend_amount, # int
                'cross_val_num': self.cross_val_num, # int
                # these need to be updated each time:
                'cpg_ids': cpg_ids, # numpy array
                'feat_mats': feat_mats, # dict of sparse numpy arrays
                'feat_names': feat_names, # dict of lists
                'target_values': target_values, # dict of pandas series
                'mad_target_values': mad_target_values # dict of pandas series
                }
        # if mutation_features_store is not empty, add to it
        else: 
            self.mutation_features_store['feat_mats'].update(feat_mats)
            self.mutation_features_store['feat_names'].update(feat_names)
            self.mutation_features_store['target_values'].update(target_values)
            self.mutation_features_store['mad_target_values'].update(mad_target_values)
            # append to cpg_ids numpy.ndarray
            self.mutation_features_store['cpg_ids'] = np.append(self.mutation_features_store['cpg_ids'], cpg_ids)
      
      
    def choose_cpgs_to_train(
        self,
        bin_size: int = 50000,
        sort_by: list = ['count', 'abs_age_corr']
        ) -> pd.DataFrame:
        """
        Based on count of mutations nearby and mutual information, choose cpgs to train models for
        @ metric_df: dataframe of mutual information or corrs
        @ bin_size: size of bins to count mutations in
        @ returns: cpg_pred_priority, dataframe of cpgs with priority for prediction
        """
        def mutation_bin_count(
            all_mut_w_age_df: pd.DataFrame
            ) -> pd.DataFrame:
            """
            Count the number of mutations in each 50kb bin across all training samples
            """
            # bin the ages
            all_mut_w_age_df['age_bin'] = pd.cut(
                all_mut_w_age_df['age_at_index'], 
                bins = np.arange(all_mut_w_age_df['age_at_index'].min(), all_mut_w_age_df['age_at_index'].max(), 10)
                )
            mut_bin_counts_dfs = []
            # for each chromosome
            for chrom in all_mut_w_age_df['chr'].unique():
                # get the mutations in the chromosome
                chr_df = all_mut_w_age_df.loc[
                    (all_mut_w_age_df['chr'] == chrom) 
                    & (all_mut_w_age_df['case_submitter_id'].isin(self.train_samples))
                    ]
                # count the number of mutations in each bin
                max_start = chr_df['start'].max()
                counts, edges = np.histogram(
                    chr_df['start'], bins = np.arange(0, max_start, bin_size)
                    )
                one_mut_bin_counts_df = pd.DataFrame({'count': counts, 'bin_edge_l': edges[:-1]})
                one_mut_bin_counts_df['chr'] = chrom  
                mut_bin_counts_dfs.append(one_mut_bin_counts_df)
                
            mut_bin_counts_df = pd.concat(mut_bin_counts_dfs, axis = 0)
            mut_bin_counts_df.reset_index(inplace=True, drop=True)
            return mut_bin_counts_df
            
        def round_down(num) -> int:
            """
            Round N down to nearest bin_size
            """
            return num - (num % bin_size)
        
        mutation_bin_counts_df = mutation_bin_count(self.all_mut_w_age_df)
        # select the cpgs with methylation data
        illumina_cpg_locs_w_methyl_df = self.illumina_cpg_locs_df.loc[
            self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)
            ]
        # round down the start position to the nearest bin_size
        illumina_cpg_locs_w_methyl_df.loc[:,'rounded_down_start'] = illumina_cpg_locs_w_methyl_df.loc[:, 'start'].apply(round_down)
        # merge the cpgs with the mutation counts
        cpg_pred_priority_down = illumina_cpg_locs_w_methyl_df[['#id', 'chr', 'rounded_down_start']].merge(
            mutation_bin_counts_df, left_on=['chr', 'rounded_down_start'],
            right_on=['chr', 'bin_edge_l'], how='left'
            )
        # also round up the start position to the nearest bin_size
        # this is to let CpGs get a high nearby mut count upstream or downstream
        illumina_cpg_locs_w_methyl_df['rounded_up_start'] = illumina_cpg_locs_w_methyl_df['rounded_down_start'] + bin_size
        # and merge this too
        cpg_pred_priority_up = illumina_cpg_locs_w_methyl_df[['#id', 'chr', 'rounded_up_start']].merge(
            mutation_bin_counts_df, left_on=['chr', 'rounded_up_start'],
            right_on=['chr', 'bin_edge_l'], how='left'
            )
        # merge the two on cpg id
        cpg_pred_priority = cpg_pred_priority_down.merge(
            cpg_pred_priority_up, on='#id', how='outer', suffixes=('_down', '_up')
            )
        # sum the counts
        cpg_pred_priority['count'] = cpg_pred_priority['count_down'] + cpg_pred_priority['count_up'] 
        # calculate age correlation
        age_corr = self.all_methyl_age_df_t.loc[self.train_samples].corrwith(
            self.all_methyl_age_df_t.loc[self.train_samples, 'age_at_index']
            )
        age_corr.drop(['age_at_index', 'gender_MALE', 'gender_FEMALE'], inplace=True)
        age_corr = age_corr.to_frame()
        age_corr.columns = ['age_corr']
        age_corr['abs_age_corr'] = age_corr['age_corr'].abs()
        # combine with age corr
        cpg_pred_priority = cpg_pred_priority.merge(age_corr, left_on='#id', right_index=True, how='left')
        # sort by count and age corr
        cpg_pred_priority.sort_values(by=sort_by, ascending=[False, False], inplace=True)
        # drop na and reset index
        cpg_pred_priority.dropna(inplace=True, how='any', axis=0)
        cpg_pred_priority.reset_index(inplace=True, drop=True)
        return cpg_pred_priority
          
    def save_mutation_features(
        self,
        start_top_cpgs: str = ""
        ) -> None:
        """
        Write out the essential data as a dictionary to a file in a directory
        """
        # create file name based on mutation_features_store meta values
        meta_str = self.consortium + '_' + self.mutation_features_store['dataset'] + '_' + str(self.mutation_features_store['num_correl_sites']) + 'correl_' + str(self.mutation_features_store['max_meqtl_sites']) + 'meqtl_'+ str(self.mutation_features_store['nearby_window_size']) + 'nearby_' + str(self.mutation_features_store['aggregate']) + 'agg_' + str(len(self.mutation_features_store['cpg_ids'])) + 'numCpGs_' + str(start_top_cpgs) + 'startTopCpGs_'  + str(self.mutation_features_store['extend_amount']) + 'extendAmount_' + str(self.mutation_features_store['cross_val_num']) + 'crossValNum'
        
        # create directory if it doesn't exist
        directory = os.path.join(self.out_dir, meta_str)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # pickle self.mutation_features_store
        fn = os.path.join(self.out_dir, meta_str, meta_str + '.features.pkl')
        with open(fn, 'wb') as f:
            pickle.dump(self.mutation_features_store, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved mutation features to\n' + fn, flush=True)
        return fn
        
    def load_mutation_features(
        self, 
        fn: str
        ) -> None:
        """
        Load in self.mutation_features_store from a pickled file in a directory
        """
        # read in the pickled file
        with open(fn, 'rb') as f:
            self.mutation_features_store = pickle.load(f)
        # set the class attributes based on the loaded data
        self.dataset = self.mutation_features_store['dataset']
        self.train_samples = self.mutation_features_store['train_samples']
        self.test_samples = self.mutation_features_store['test_samples']
        self.cpg_ids = self.mutation_features_store['cpg_ids']
        self.aggregate = self.mutation_features_store['aggregate']
        
        
