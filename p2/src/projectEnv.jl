""" Módulo fachada para facilitar importaciones """

module ProjectEnv

    # =========================================================================
    # CARGAR ARCHIVOS FUENTE
    # =========================================================================
    
    include("scaler.jl")
    include("filterWrappers.jl")
    include("dimReduction.jl")
    include("modelFactory.jl")
    
    include("experimentLab.jl")
    include("hypothesisTest.jl")
    
    include("resultAnalysis.jl")

    # =========================================================================
    # IMPORTACIÓN
    # =========================================================================
    using .CustomScalers
    using .FilteringReduction
    using .ProyectionReduction
    using .ModelFactory
    using .ExperimentLab
    using .ResultAnalysis
    using .StatisticalTests

    # =========================================================================
    # EXPORTACÓN
    # =========================================================================

    # --- filterWrappers ---
    export PearsonSelector, SpearmanSelector, KendallSelector, 
           ANOVASelector, MutualInfoSelector, RFELogistic

    # --- dimReduction ---
    export get_lda_model, get_pca_model, get_ica_model

    # --- modelFactory ---
    export get_knn_model, get_svm_model, get_mlp_model, 
           get_bagging_knn_model, get_evotree_model, get_adaboost_model, 
           get_rf_model, get_xgboost_model, get_lightgbm_model, 
           get_catboost_model, get_voting_classifier, get_stacking_model, 
           get_individual_wise_CV, IndividualWiseCV

    # --- experimentLab ---
    export run_experiment_crossvalidation, run_experiment_holdout, History

    # --- scaler ---
    export MinMaxScaler

    # --- resultAnalysis ---
    export display_cv_table, plot_cv_results, display_holdout_table, 
           display_confussion_matrix, process_feature_importance

    # --- hypothesisTest ---
    export auto_compare_models

end