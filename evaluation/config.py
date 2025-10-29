# evaluation/config.py

"""
Define evaluation suites by mapping a suite name to a list of game_ids.
"""

EVALUATION_GAMES = {
    # A small suite for quick debugging and testing
    "debug_suite": [
        # "ft09-b8377d4b7815",
        "as66-821a4dcad9c2",
    ],
    # A more comprehensive suite for full evaluations
    "standard_suite": [
        "as66-821a4dcad9c2",
        "ls20-fa137e247ce6",
        "ft09-b8377d4b7815",
        "vc33-6ae7bf49eea5",
        "lp85-d265526edbaa",
        "sp80-0605ab9e5b2a",
    ],
}
