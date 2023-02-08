import pandas as pd
import numpy as np
import sys
import os
import pickle
import ray


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
        dataset: str,
        train_samples: list,
        test_samples: list,
        matrix_qtl_dir: str
        ):
        self.all_mut_w_age_df = all_mut_w_age_df
        self.illumina_cpg_locs_df = illumina_cpg_locs_df
        self.all_methyl_age_df_t = all_methyl_age_df_t
        self.out_dir = out_dir
        self.dataset = dataset
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.matrix_qtl_dir = matrix_qtl_dir
        # create empty cache
        self.matrixQTL_store = {}
        # pre-process the mutation and methylation data
        self._preproc_mut_and_methyl()
        self.all_samples = self.all_methyl_age_df_t.index.to_list()
    
    def _preproc_mut_and_methyl(
        self
        ) -> None:
        """
        Create all_mut_w_age_illum_df, remove X & Y chromosomes,
        and only keep samples with measured methylation
        """
        # if a dataset is specified, subset the data to only this tissue type
        if self.dataset != "":
            self.all_methyl_age_df_t = self.all_methyl_age_df_t.loc[
                self.all_methyl_age_df_t['dataset'] == self.dataset, :]
            self.all_mut_w_age_df = self.all_mut_w_age_df.loc[
                self.all_mut_w_age_df['dataset'] == self.dataset, :].copy(deep=True)
        # if a mut_loc column does not exit, add it
        if 'mut_loc' not in self.all_mut_w_age_df.columns:
            self.all_mut_w_age_df['mut_loc'] = self.all_mut_w_age_df['chr'] + ':' \
                                                + self.all_mut_w_age_df['start'].astype(str)
        # only non X and Y chromosomes and that occured in samples with measured methylation
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
            set(self.all_methyl_age_df_t.columns).intersection(set(self.illumina_cpg_locs_df['#id'].to_list() + ['dataset', 'gender', 'age_at_index']))
            ]
        # one hot encode covariates
        if self.dataset == "":
            dset_col = self.all_methyl_age_df_t['dataset'].to_list()
            self.all_methyl_age_df_t = pd.get_dummies(self.all_methyl_age_df_t, columns=["gender", "dataset"])
            # add back in the dataset column
            self.all_methyl_age_df_t['dataset'] = dset_col
        else: # only do gender if one dataset is specified
            self.all_methyl_age_df_t = pd.get_dummies(self.all_methyl_age_df_t, columns=["gender"])

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
                os.path.join(self.matrix_qtl_dir, f"chr{chrom}_meqtl.parquet"),
                columns=['#id', 'SNP', 'p-value', 'beta'])            
            self.matrixQTL_store[chrom] = meqtl_df
        else:
            meqtl_df = self.matrixQTL_store[chrom]
        # get the meQTLs for this CpG
        neg_meqtls = meqtl_df.loc[(meqtl_df['#id'] == cpg_id) & (meqtl_df['beta'] < 0), :].nsmallest(max_meqtl_sites, 'p-value')['SNP'].to_list()
        pos_meqtls = meqtl_df.loc[(meqtl_df['#id'] == cpg_id) & (meqtl_df['beta'] > 0), :].nsmallest(max_meqtl_sites, 'p-value')['SNP'].to_list()
        return neg_meqtls, pos_meqtls
    
    def _get_predictor_site_groups(
        self, 
        cpg_id: str,
        num_correl_sites: int = 5000,
        num_correl_ext_sites: int = 100,
        max_meqtl_sites: int = 100,
        nearby_window_size: int = 25000
        ) -> list:
        """
        Get the sites to be used as predictors of cpg_id's methylation
        @ cpg_id: the id of the CpG
        @ num_correl_sites: the number of correlated sites to be used
        @ max_meqtl_sites: the maximum number of meQTLs to be used
        @ nearby_window_size: the window size to be used to find nearby sites
        @ returns: dict of types of genomic locations of the sites to be used as predictors in format chr:start
        """
        def extend(loc_list):
            """
            Return the list of locations with each original loc extended out 250bp in each direction
            """
            extend_amount = 250
            return [loc.split(':')[0] + ':' + str(int(loc.split(':')[1]) + i) 
                    for loc in loc_list 
                    for i in range(-extend_amount, extend_amount + 1)]
        
        predictor_site_groups = {}
        try: # get cpg_id's chromosome and start position
            chrom = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'] == cpg_id, 'chr'].values[0]
            start = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'] == cpg_id, 'start'].values[0]
        except: # for some reason this cpg is not in illumina_cpg_locs_df, return empty dict
            return {}
        # get dict of positions of most positively and negatively correlated CpGs
        predictor_site_groups['pos_corr'], predictor_site_groups['neg_corr'] = self._select_correl_sites(
            cpg_id, chrom, num_correl_sites=num_correl_sites)
        # get extended versions of the num_correl_ext_sites most positively and negatively correlated CpGs
        predictor_site_groups['pos_corr_ext']= extend(
            predictor_site_groups['pos_corr'][:num_correl_ext_sites]
            )
        predictor_site_groups['neg_corr_ext'] = extend(
            predictor_site_groups['neg_corr'][:num_correl_ext_sites]
            )
        # get nearby sites
        predictor_site_groups['nearby'] = [
            chrom + ':' + str(start + i)
            for i in range(-int(nearby_window_size/2), int(nearby_window_size/2) + 1)
            ]
        # get sites from matrixQTL 
        predictor_site_groups['matrixqtl_neg_beta'], predictor_site_groups['matrixqtl_pos_beta'] = self._get_matrixQTL_sites(cpg_id, chrom, max_meqtl_sites=max_meqtl_sites)
        return predictor_site_groups
    
    def _create_feature_mat(
        self, 
        cpg_id: str, 
        predictor_groups: dict,
        aggregate: str
        ) -> tuple:
        """
        Create the training matrix for the given cpg_id and predictor_sites
        @ cpg_id: the id of the CpG
        @ predictor_groups: dict of lists of sites to be used as predictors of cpg_id's methylation
        @ aggregate: whether to aggregate the mutation status by predictor group
        @ returns: X, y where X is the feature matrix and y is the methylation values of cpg_id across samples
        """
        # make list of all samples to fill in samples that do not have mutations in predictor sites
        all_samples = self.train_samples + self.test_samples
        
        def noAgg():
            """
            Create a feature matrix where each column is a predictor site from predictor_groups
            and values are the variant allele frequencies of the mutations at that site in the sample
            """
            # get list of unique predictor sites
            predictor_sites = set()
            for key in predictor_groups:
                predictor_sites.update(predictor_groups[key])
            predictor_sites = list(predictor_sites)
            # get the mutation status of predictor sites
            mut_status = self.all_mut_w_age_df.loc[
                self.all_mut_w_age_df['mut_loc'].isin(predictor_sites),
                ['DNA_VAF', 'case_submitter_id', 'mut_loc']
                ]
            # create a new dataframe with columns = predictor sites, rows = y.index,
            # and values = variant allele frequencies
            feat_mat = pd.pivot_table(mut_status, index='case_submitter_id', columns='mut_loc', values='DNA_VAF', fill_value = 0)
            # add rows of all 0s for samples that don't have any mutations in predictor sites
            feat_mat = feat_mat.reindex(all_samples, fill_value=0)
            return feat_mat
        
        def agg():
            """
            Create a feature matrix where each column is a predictor group from predictor_groups
            and values are the sum of the variant allele frequencies of the mutations in
            that group in the sample
            """
            aggregated_muts = []
            for _, one_group_sites in predictor_groups.items():
                mut_status = self.all_mut_w_age_df.loc[
                    self.all_mut_w_age_df['mut_loc'].isin(one_group_sites),
                    ['DNA_VAF', 'case_submitter_id', 'mut_loc']
                    ]
                mut_status = pd.pivot_table(
                    mut_status, index='case_submitter_id', columns='mut_loc',
                    values='DNA_VAF', fill_value = 0
                    )
                mut_status = mut_status.reindex(all_samples, fill_value=0)
                agg_mut_status = mut_status.sum(axis=1)
                aggregated_muts.append(agg_mut_status)
            # create dataframe from aggreated mutations with columns pred_type
            agg_feat_mat = pd.concat(aggregated_muts, axis=1)
            agg_feat_mat.columns = predictor_groups.keys()
            return agg_feat_mat
        
        if aggregate == "Both":
            feat_mat = noAgg()
            agg_feat_mat = agg()
            feat_mat = pd.merge(feat_mat, agg_feat_mat, left_index=True, right_index=True)
        elif aggregate == "False":
            feat_mat = noAgg()
        elif aggregate == "True":
            feat_mat = agg()
        else:
            sys.exit("Aggregate must be either 'True', 'False', or Both")
        # add covariate columns to X
        coviariate_col_names = [
            col for  col in self.all_methyl_age_df_t.columns \
            if col.startswith('dataset_') or col.startswith('gender_')
            ]
        covariate_df = self.all_methyl_age_df_t.loc[all_samples, coviariate_col_names]
        feat_mat = pd.merge(feat_mat, covariate_df, left_index=True, right_index=True)
        y = self.all_methyl_age_df_t.loc[all_samples, cpg_id]
        return feat_mat, y
    
    def choose_cpgs_to_train(
        self,
        mi_df: pd.DataFrame,
        bin_size: int = 20000
        ) -> pd.DataFrame:
        """
        Based on count of mutations nearby and mutual information, choose cpgs to train models for
        @ mi_df: dataframe of mutual information
        @ bin_size: size of bins to count mutations in
        @ returns: cpg_pred_priority, dataframe of cpgs with priority for prediction
        """
        def mutation_bin_count(
            all_mut_w_age_df: pd.DataFrame
            ) -> pd.DataFrame:
            """
            Count the number of mutations in each 20kb bin across all training samples
            """
            mut_bin_counts_dfs = []
            for chrom in all_mut_w_age_df['chr'].unique():
                chr_df = all_mut_w_age_df.loc[(all_mut_w_age_df['chr'] == chrom) & (all_mut_w_age_df['case_submitter_id'].isin(self.train_samples))]
                counts, edges = np.histogram(chr_df['start'], bins = np.arange(0, chr_df['start'].max(), bin_size))
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
        
        # count mutations in each bin
        mutation_bin_counts_df = mutation_bin_count(self.all_mut_w_age_df)
        # get count for each cpg
        illumina_cpg_locs_w_methyl_df = self.illumina_cpg_locs_df.loc[self.illumina_cpg_locs_df['#id'].isin(self.all_methyl_age_df_t.columns)]
        illumina_cpg_locs_w_methyl_df.loc[:,'rounded_start'] = illumina_cpg_locs_w_methyl_df.loc[:, 'start'].apply(round_down)
        cpg_pred_priority = illumina_cpg_locs_w_methyl_df.merge(mutation_bin_counts_df, left_on=['chr', 'rounded_start'], right_on=['chr', 'bin_edge_l'], how='left')
        # get mi for each cpg
        cpg_pred_priority = cpg_pred_priority.merge(mi_df, left_on='#id', right_index=True, how='left')
        # sort by count and mi
        cpg_pred_priority.sort_values(by=['count', 'mutual_info'], ascending=[False, False], inplace=True)
        # reset index
        cpg_pred_priority.reset_index(inplace=True, drop=True)
        return cpg_pred_priority
    
    def create_all_feat_mats(
        self, 
        cpg_ids: list, 
        aggregate: str,
        num_correl_sites: int = 5000,
        num_correl_ext_sites: int = 100,
        max_meqtl_sites: int = 100,
        nearby_window_size: int = 25000
        ) -> dict:
        """
        Create the training matrix for the given cpg_id and predictor_sites
        @ cpg_ids: the ids of the CpGs
        @ predictor_groups: dict of lists of sites to be used as predictors of cpg_id's methylation
        @ samples: the samples to be included in the training matrix
        @ aggregate: whether to aggregate the mutation status by predictor group
        @ returns: X, y where X is the training matrix and y is the methylation values of cpg_id across samples
        """
        feat_mats = {}
        target_values = {}
        for cpg_id in cpg_ids:
            # first get the predictor groups
            predictor_groups = self._get_predictor_site_groups(
                cpg_id, num_correl_sites, num_correl_ext_sites, 
                max_meqtl_sites, nearby_window_size
                )
            # then create the feature matrix from these
            feat_mats[cpg_id], target_values[cpg_id] = self._create_feature_mat(
                cpg_id, predictor_groups, aggregate
                )
        # create a dictionary to allow for easy data persistence
        self.mutation_features_store = {
            'dataset': self.dataset,
            'train_samples': self.train_samples,
            'test_samples': self.test_samples,
            'cpg_ids': cpg_ids,
            'aggregate': aggregate,
            'num_correl_sites': num_correl_sites,
            'num_correl_ext_sites': num_correl_ext_sites,
            'max_meqtl_sites': max_meqtl_sites,
            'nearby_window_size': nearby_window_size,
            'feat_mats': feat_mats,
            'target_values': target_values
            }
            
    def save_mutation_features(
        self
        ) -> None:
        """
        Write out the essential data as a dictionary to a file in a directory
        """
        # create file name based on mutation_features_store meta values
        fn = os.path.join(self.out_dir, 
                            self.mutation_features_store['dataset'] + '_'
                            + str(self.mutation_features_store['num_correl_sites']) + 'correl_'
                            + str(self.mutation_features_store['num_correl_ext_sites']) + 'correlExt_'
                            + str(self.mutation_features_store['max_meqtl_sites']) + 'meqtl_'
                            + str(self.mutation_features_store['nearby_window_size']) + 'nearby_'
                            + str(self.mutation_features_store['aggregate']) + 'agg_'
                            + str(len(self.mutation_features_store['cpg_ids'])) + 'numCpGs.features'
                            )
        # pickle self.mutation_features_store
        with open(fn, 'wb') as f:
            pickle.dump(self.mutation_features_store, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved mutation features to\n' + fn)
        
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
        
        