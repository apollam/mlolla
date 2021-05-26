from mlolla.data.transformers import ColumnDropper


def test_simple_drop(input_to_drop):
    transformer = ColumnDropper(columns_to_drop=['A'])
    data = transformer.fit_transform(input_to_drop)
    assert set(data.columns.tolist()) == set(['B', 'C', 'B_correlated', 'single'])
    assert transformer.columns_to_drop == ['A']


def test_specific_correlation_drop(input_to_drop):
    transformer = ColumnDropper(correlation_to_drop={'B': 0.3})
    data = transformer.fit_transform(input_to_drop)
    assert set(data.columns.tolist()) == set(['A', 'B', 'C', 'single'])
    assert transformer.columns_to_drop == ['B_correlated']


def test_correlation_drop(input_to_drop):
    transformer = ColumnDropper(correlation_to_drop=0.5)
    data = transformer.fit_transform(input_to_drop)
    print(data)
    assert set(data.columns.tolist()) == set(['A', 'single'])
    assert transformer.columns_to_drop == ['B', 'B_correlated', 'C']


def test_missing_drop(input_to_drop):
    transformer = ColumnDropper(missing_threshold=0.3)
    data = transformer.fit_transform(input_to_drop)
    assert set(data.columns.tolist()) == set(['B', 'C', 'B_correlated', 'single'])
    assert transformer.columns_to_drop == ['A']


def test_single_value_drop(input_to_drop):
    transformer = ColumnDropper(drop_uniques_values=True)
    data = transformer.fit_transform(input_to_drop)
    assert set(data.columns.tolist()) == set(['A', 'B', 'C', 'B_correlated'])
    assert transformer.columns_to_drop == ['single']