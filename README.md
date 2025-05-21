# KAC Assignment Problem

This tool creates equally sized teams for KAC events where participants can specify their preferences for teammates.

It begins by constructing an **adjacency matrix** representing these preferences. The entry $x_{ij}$ is 1 if either person $i$, person $j$, or both wish to be on the same team.

Next, the tool uses an **Integer Linear Programming (ILP)** approach to maximize overall team happiness based on this adjacency matrix.

Finally, it outputs a CSV file detailing the team allocations.

---

## Installation

To install, using `pip`:

```bash
pip install git+https://github.com/Samwurel/KACAssignment
```

Alternatively, you can clone the repository and install it:

```bash
git clone https://github.com/Samwurel/KACAssignment KACAssignment
cd KACAssignment
pip install .
```

---

## Usage

```bash
# Example 1: Select the CSV file, set team size to 6, and include a test row, preference column name: "Preference", write to file
# .\data\out.csv
kacassignment path\\to\\file.csv -t 6 --testrow -p Preference -o .\\data\\out.csv

# Example 2: Select the CSV file, specify number of teams as 8, and exclude the test row
kacassignment path\\to\\file.csv -n 8 --no-testrow
```

## Arguments:
* `-t, --team-size <int>`: Number of people per team
* `-n, --num-teams <int>`: Number of teams to form
* `--testrow / --no-testrow`: Whether to keep the test/form response row (default: `--testrow`)
* `-p, --preference`: The name of the column in which preference names are stored (default: "Preference")
* `--help`: Display the help message
* `--version`: Display the version information
* `--verbose`: Enable verbose output
---

## Data Format

The input should be a standard KAC signup form response sheet that includes a **"Preference"** column.

### Preparing the Preference Column

When creating or editing the **Preference** column, make sure to:

* Use correct spelling for all names to ensure accurate matching.
* Separate multiple names with commas (e.g., `Alice Smith, Bob Jones`).
* Leave the cell **blank** if there are no preferences — delete values like `"None"` or `"N/A"`.

This column is used to guide team assignments based on mutual preferences. Proper formatting improves the quality of the results.

---

## License

kacassignment is licensed under the [MIT License](LICENSE).

