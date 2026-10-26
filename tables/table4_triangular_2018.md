# Triangular-demand comparison with the 2018 cubic MINLP

| Formulation | Metric | Statistic | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Big-M (9), Gurobi | CPU Sec | Avg | 7.206E-01 | 1.525E+00 | 2.237E+00 | 1.703E+00 | 3.136E+00 | 3.775E+00 |
| Big-M (9), Gurobi | CPU Sec | Avg inc lim | 7.206E-01 | 1.525E+00 | 2.237E+00 | 1.703E+00 | 3.136E+00 | 3.775E+00 |
| Big-M (9), Gurobi | B&B Nodes | Avg | 7.417E+01 | 8.787E+01 | 1.126E+02 | 4.323E+01 | 1.050E+02 | 8.397E+01 |
| Big-M (9), Gurobi | B&B Nodes | Avg inc lim | 7.417E+01 | 8.787E+01 | 1.126E+02 | 4.323E+01 | 1.050E+02 | 8.397E+01 |
| Big-M (9), Gurobi | Time Limit | Number | 0 | 0 | 0 | 0 | 0 | 0 |
| Perspective, Gurobi | CPU Sec | Avg | 6.856E-01 | 1.863E+00 | 2.040E+00 | 1.584E+00 | 2.425E+00 | 2.494E+00 |
| Perspective, Gurobi | CPU Sec | Avg inc lim | 6.856E-01 | 1.863E+00 | 2.040E+00 | 1.584E+00 | 2.425E+00 | 2.494E+00 |
| Perspective, Gurobi | B&B Nodes | Avg | 8.430E+01 | 1.354E+02 | 1.080E+02 | 5.347E+01 | 9.957E+01 | 5.620E+01 |
| Perspective, Gurobi | B&B Nodes | Avg inc lim | 8.430E+01 | 1.354E+02 | 1.080E+02 | 5.347E+01 | 9.957E+01 | 5.620E+01 |
| Perspective, Gurobi | Time Limit | Number | 0 | 0 | 0 | 0 | 0 | 0 |
| 2018 cubic MINLP, SCIP | CPU Sec | Avg | 2.567E+02 | 2.105E+02 | 8.600E+01 | 7.200E+01 | 1.407E+02 | 6.830E+02 |
| 2018 cubic MINLP, SCIP | CPU Sec | Avg inc lim | 9.253E+02 | 2.357E+03 | 2.897E+03 | 2.777E+03 | 2.562E+03 | 3.211E+03 |
| 2018 cubic MINLP, SCIP | B&B Nodes | Avg | 1.040E+04 | 4.881E+03 | 4.875E+02 | 4.329E+01 | 2.366E+02 | 7.774E+03 |
| 2018 cubic MINLP, SCIP | B&B Nodes | Avg inc lim | 5.748E+04 | 5.773E+04 | 4.071E+04 | 3.282E+04 | 2.919E+04 | 3.063E+04 |
| 2018 cubic MINLP, SCIP | Time Limit | Number | 6 | 19 | 24 | 23 | 21 | 26 |
