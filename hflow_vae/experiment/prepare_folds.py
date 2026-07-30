from sklearn.model_selection import StratifiedKFold, train_test_split
from data.data_processor import TabularDataProcessor
from experiment.data_objects import FoldData


def prepare_folds(
    X,
    y,
    dataset_name,
    datasets_info,
    seed,
    fold_threshold=300,
    val_ratio=0.15,
    num_transform_method="quantile",
    cat_encoding_method="ordinal",
):


    dataset_info = datasets_info[dataset_name]

    n_splits = 3 if len(X) < fold_threshold else 5

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        # ====================
        # Train/Test
        # ====================
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()

        X_test  = X.iloc[test_idx].copy()
        y_test  = y.iloc[test_idx].copy()

        # ====================
        # Train/Validation
        # ====================
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=val_ratio,
            stratify=y_train,
            random_state=seed,
        )

        # ====================
        # Processor
        # ====================
        processor = TabularDataProcessor(
            num_cols=dataset_info["numerical"],
            cat_cols=dataset_info["categorical"],
            num_transform_method=num_transform_method,
            cat_encoding_method=cat_encoding_method,
            random_state=seed,
        )

        processor.fit(X_train)

        if dataset_info["has_missing"]:
            X_train = processor.transform_readable(X_train)
            X_valid = processor.transform_readable(X_valid)
            X_test  = processor.transform_readable(X_test)

        train_pos_mask = (y_train == 1)
        valid_pos_mask = (y_valid == 1)
        test_pos_mask = (y_test == 1)

        yield FoldData(

            dataset=dataset_name,
            seed=seed,
            fold=fold,

            processor=processor,

            num_cols=dataset_info["numerical"],
            cat_cols=dataset_info["categorical"],
            target_col = dataset_info["target"],
            has_missing = dataset_info["has_missing"],
            
            X_train     = X_train,
            y_train     = y_train,
            X_train_pos = X_train[train_pos_mask],
            y_train_pos = y_train[train_pos_mask],

            X_valid     = X_valid,
            y_valid     = y_valid,
            X_valid_pos = X_valid[valid_pos_mask],
            y_valid_pos = y_valid[valid_pos_mask],

            X_test      = X_test,
            y_test      = y_test,
            X_test_pos  = X_test[test_pos_mask],
            y_test_pos  = y_test[test_pos_mask],
        )