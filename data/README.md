# Instances

This directory contains ten synthetic UAV--UGV two-echelon distribution
instances used in Figure 1. Each instance consists of:

- `*_nodes.csv`: one fulfillment center, candidate hubs, and customers;
- `*_params.csv`: capacity, endurance, speed, setup-time, and cost parameters.

`case_matrix.csv` expands these ten physical instances into the twenty released
experiment configurations: ten commercial and ten emergency configurations.

The featured instance is `instance_20260103` (55 customers and 7 candidate
hubs). Node identifiers and coordinates are synthetic research data and do not
contain customer names, addresses, or other personal information.

The emergency profile is created at run time by setting the hub, UAV, and UGV
capital-cost coefficients to zero. Physical data and running-cost coefficients
remain unchanged.
