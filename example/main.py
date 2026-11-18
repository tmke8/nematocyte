import nematocyte

from src.run import Config

# This is the main entry point for the script.
# Meaning of the arguments to @nematocyte.main():
#     Config: The dataclass that defines which keys and values are valid in the
#         configuration. The function below receives an instance of it.
#     config_path: The path to the directory containing the yaml config files,
#         relative to this file.
# There is no default config file: if you run the script without `--config-name`,
# all values come from the defaults in the `Config` class.


@nematocyte.main(Config, config_path="configs")
def main(config: Config) -> float:
    # Set up Python's `logging` module: messages go to standard output and to
    # `run.log` in the output directory of the run. The level comes from the
    # setting `nematocyte.log`, which is `info` unless you change it, for
    # example with `nematocyte.log=debug` on the command line.
    nematocyte.configure_logging()

    # `run_config` returns the config of this run as plain dicts, lists, strings
    # and numbers, which is what we want to send to the logger. It is exactly
    # what nematocyte saves in `.nematocyte/config.json` in the output directory.
    config_for_logging = nematocyte.run_config()

    # Finally, we call the `run` method of the `Config` object.
    return config.run(config_for_logging)


if __name__ == "__main__":
    main()
