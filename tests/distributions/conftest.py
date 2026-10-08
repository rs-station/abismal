def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: takes tens of seconds (e.g. torch.compile); deselect with -m 'not slow'"
    )
