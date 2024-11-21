
"""
CELL GREP SINGLE
Mostly just for testing
"""


import cell_grep
from cell_grep import UserError


def main():
    logger = None
    try:
        args = parse_args()
        logger = cell_grep.setup_logging(args)
        table  = cell_grep.read_table(args)
        logger.info("table size: %i", len(table))

        index = cell_grep.make_index(table)

    except UserError as e:
        msg = "cell_grep: user error: " + " ".join(e.args)
        if logger is None:
            print(msg)
        else:
            logger.fatal()
        exit(1)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        prog="cell-grep-single",
        description=
        """
        Select keys from GDSC that match or don't.
        The pattern is for a simple substring match.
        Input: table.tsv Output: selection.txt
        """)
    parser.add_argument("input_data",
                        help="The main GDSC CSV file")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
