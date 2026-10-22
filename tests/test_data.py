from src.data import load_all_sites, load_pooled_data, get_site_summary, SITE_NAMES, NUM_SITES

def test_site_metadata():
    assert NUM_SITES == 4
    assert SITE_NAMES == ["cleveland", "hungarian", "switzerland", "va"]

def test_load_all_sites():
    sites_data = load_all_sites()
    assert len(sites_data) == 4
    for sid in range(NUM_SITES):
        assert sid in sites_data
        train_x, train_y = sites_data[sid]["train"]
        test_x, test_y = sites_data[sid]["test"]

        assert train_x.shape[1] == 13, f"Expected 13 tabular features, got {train_x.shape[1]}"
        assert test_x.shape[1] == 13
        assert train_x.shape[0] == train_y.shape[0]
        assert test_x.shape[0] == test_y.shape[0]

def test_pooled_data():
    x_tr, y_tr = load_pooled_data(train=True)
    x_te, y_te = load_pooled_data(train=False)

    assert x_tr.shape == (486, 13)
    assert y_tr.shape == (486, 1)
    assert x_te.shape == (254, 13)
    assert y_te.shape == (254, 1)

def test_site_summary():
    df = get_site_summary()
    assert len(df) == 4
    assert "n_train" in df.columns
    assert "n_test" in df.columns
    assert "prev_train" in df.columns
    assert df["n_total"].sum() == 740
