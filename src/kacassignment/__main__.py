import pandas as pd
import numpy as np
import os
import sys
import click

from pathlib import Path
from typing import List, Dict, Any


@click.command()
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
def main(file_path: Path, team_size: int | None, num_teams: int | None, testrow: bool) -> None:
    """
    Main function to create teams based on preferences from a CSV file.

    Args:
        file_path (Path): Path to the CSV file containing preferences.
        team_size (int | None): Size of the teams to be created.
        num_teams (int | None): Number of teams to be created.
        testrow (bool): If True, drop the first row of the DataFrame.
    
    Raises:
        click.UsageError: If both team_size and num_teams are provided or neither is provided.
    
    Returns:
        None
    """
    if team_size is None and num_teams is None:
        raise click.UsageError(
            "You must provide either --num-teams or --team-size as an argument, not both."
        )
    elif team_size is not None and num_teams is not None:
        raise click.UsageError(
            "You must provide either --num-teams or --team-size as an argument, not both."
        )


    data = parse_csv(Path(file_path), testrow = testrow)
    print(data.head(10))

    matrix = preference_matrix(data)


from pathlib import Path
import pandas as pd

def parse_csv(path: Path, testrow: bool) -> pd.DataFrame:
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
    for col in ["First name", "Last name", "Preference"]:
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

def make_teams(preferences: List[int], data: pd.DataFrame, team_size: int) -> pd.DataFrame:
    """
    Create teams using an LP based on the preference matrix and team size.

    Args:
        preferences (List[int]): Preference matrix as a list of lists.
        data (pd.DataFrame): DataFrame containing the preference data.
        team_size (int): Size of the teams to be created.

    Returns:
        pd.DataFrame: DataFrame containing the teams.
    """
    
    # Go throught the peferences

    
if __name__ == "__main__":
    main()
