import pandas as pd
import numpy as np
import os
import sys
import click

from pathlib import Path
from pulp import LpProblem, LpMaximize, LpVariable, LpStatus, LpBinary, LpStatusOptimal, lpSum, PULP_CBC_CMD
from typing import List, Dict, Any
from importlib.metadata import version

import threading
import time
from tqdm import tqdm

package = __package__

@click.command()
@click.version_option(version=version(package), prog_name=package)
@click.argument(
    "file_path",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option(
    "--team-size",
    "-t",
    "team_size",
    required=False,
    help="The size of the teams to be created.",
    type=int,
)
@click.option(
    "--num-teams",
    "-n",
    "num_teams",
    required=False,
    help="The number of teams to be created.",
    type=int,
)
@click.option(
    "--testrow/--no-testrow",
    "testrow",
    default = True,
    help = "If the first row is a test entry."
)
@click.option(
    "--verbose",
    "verbose",
    default = 0,
    help = "Level of verbosity.",
    type=int,
)
@click.option(
    "--preference",
    "-p",
    "pref_col",
    default = "Preference",
    help = "The name of the column containing preferences.",
    type=str,
)
@click.option(
    "--output",
    "-o",
    "output",
    default = "final_teams.csv",
    help = "The name of the output CSV file.",
    type=str,
)
def main(file_path: Path, team_size: int | None, num_teams: int | None, testrow: bool, verbose: int, pref_col: str, output: str) -> None:
    """
    Main function to create teams based on preferences from a CSV file.

    Args:
        file_path (Path): Path to the CSV file containing preferences.
        team_size (int | None): Size of the teams to be created.
        num_teams (int | None): Number of teams to be created.
        testrow (bool): If True, drop the first row of the DataFrame.
        preference_colname (str): Name of the column containing preferences.
    
    Raises:
        click.UsageError: If both team_size and num_teams are provided or neither is provided.
    
    Returns:
        None
    """
    if verbose > 1 or verbose < 0:
        click.echo("Verbose level must be either 0 or 1. Defaulting to 0.")
        verbose = 0

    if team_size is None and num_teams is None:
        raise click.UsageError(
            "You must provide either --num-teams or --team-size as an argument, not both."
        )
    elif team_size is not None and num_teams is not None:
        raise click.UsageError(
            "You must provide either --num-teams or --team-size as an argument, not both."
        )

    output_path = Path(output)
    if not output_path.parent.exists():
        raise click.UsageError(
            f"The output path {output_path.parent} does not exist."
        )

    data = parse_csv(Path(file_path), testrow = testrow, pref_col = pref_col)
    matrix = preference_matrix(data)

    # Solve the LP problem
    teams = make_teams(matrix, data, team_size=team_size, num_teams=num_teams, verbose=verbose)
    teams.to_csv(output, index=False)
    print(f"Teams have been created and saved to {output}.")


def parse_csv(path: Path, testrow: bool, pref_col: str) -> pd.DataFrame:
    """
    Parse the CSV file and return a DataFrame with the relevant columns.

    Args:
        path (Path): Path to the CSV file.
        testrow (bool): If True, drop the first row of the DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame containing the parsed data.
    """

    if not path.is_absolute():
        path = path.resolve()
    
    data = pd.read_csv(path)

    # Convert relevant columns to title case
    for col in ["First name", "Last name", pref_col]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.title()

    data["Full name"] = data["First name"] + " " + data["Last name"]
    if testrow:
        data.drop(index=0, inplace=True)
    data.drop_duplicates(subset=["Full name"], inplace=True)

    desired_columns = ["First name", "Last name", "Full name", "Preference"]

    for col in desired_columns:
        if col in data.columns:
            data[col] = data[col].str.replace(r"[\,]", ";", regex=True)
            data[col] = data[col].str.replace(r"[^\w\s\;]", "", regex=True).str.strip()

    return data[desired_columns]


def preference_matrix(data: pd.DataFrame) -> List[int]:
    """
    Create a preference matrix where each person corresponds to an index the matrix has a 1 
     in entries where the preference of a person is one way or mutual and 0 otherwise.
    
    Args:
        data (pd.DataFrame): DataFrame containing the preference data.
    
    Returns:
        List[int]: Preference matrix as a list of lists.
    """

    people = data["Full name"].tolist()
    preference_matrix = np.zeros((len(people), len(people)), dtype=int)

    for i, person in enumerate(people):
        row = data[data["Full name"] == person]
        if row.empty:
            continue

        preferences = row["Preference"].values[0].split(";")
        preferences = [p.strip() for p in preferences]

        for pref in preferences:
            matches = data[
                (data["Full name"] == pref) |
                (data["First name"] == pref) |
                (data["Last name"] == pref)
            ]

            for match_name in matches["Full name"].unique():
                if match_name in people:
                    j = people.index(match_name)
                    preference_matrix[i][j] = 1
                    preference_matrix[j][i] = 1

    return preference_matrix.tolist()

def solve_with_progress(prob, timeout=60):
    # Run solver in a separate thread
    def solve():
        prob.solve(PULP_CBC_CMD(msg=False))

    thread = threading.Thread(target=solve)
    thread.start()

    # Simulated progress bar
    with tqdm(total=timeout, desc="Solving LP", unit="s", leave=False, dynamic_ncols=True) as pbar:
        for _ in range(timeout):
            if not thread.is_alive():
                break
            time.sleep(1)
            pbar.update(1)
        thread.join()


def make_teams(preferences: List[int], data: pd.DataFrame, team_size: int, num_teams: int, verbose: int) -> pd.DataFrame:
    """
    Create teams using an LP based on the preference matrix and team size.

    Args:
        preferences (List[int]): Preference matrix as a list of lists.
        data (pd.DataFrame): DataFrame containing the preference data.
        team_size (int): Size of the teams to be created.

    Returns:
        pd.DataFrame: DataFrame containing the teams.
    """
    n = len(preferences)

    if num_teams is not None:
        team_size = int(np.ceil(n / num_teams))
    elif team_size is not None:
        num_teams = int(np.ceil(n / team_size))
    else:
        raise ValueError("Either team_size or num_teams must be provided.")

    prob = LpProblem("KAC_Team_Assignment", LpMaximize)

    # Decision variables: x[i][t] = 1 if person i is in team t
    x = [[LpVariable(f"x_{i}_{t}", cat=LpBinary) for t in range(num_teams)] for i in range(n)]

    # Auxiliary variables: y[i][j][t] = 1 if both i and j are in team t
    y = [[[LpVariable(f"y_{i}_{j}_{t}", cat=LpBinary)
           for t in range(num_teams)] for j in range(n)] for i in range(n)]

    # Objective: maximize mutual preferences on same team
    prob += lpSum(preferences[i][j] * y[i][j][t]
                  for i in range(n) for j in range(n) for t in range(num_teams) if i != j)

    # Each person in exactly one team
    for i in range(n):
        prob += lpSum(x[i][t] for t in range(num_teams)) == 1

    # Team size constraints
    for t in range(num_teams):
        prob += lpSum(x[i][t] for i in range(n)) == team_size

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for t in range(num_teams):
                prob += y[i][j][t] <= x[i][t]
                prob += y[i][j][t] <= x[j][t]
                prob += y[i][j][t] >= x[i][t] + x[j][t] - 1

    # Solve
    if verbose > 0:
        prob.solve(PULP_CBC_CMD(msg=True))
    else:
        solve_with_progress(prob)

    # Build result
    teams = []
    for t in range(num_teams):
        team_members = [data["Full name"].iloc[i] for i in range(n) if x[i][t].varValue == 1]
        for member in team_members:
            teams.append({"Team": f"Team {t+1}", "Full name": member})

    return pd.DataFrame(teams)

    
if __name__ == "__main__":
    main()
