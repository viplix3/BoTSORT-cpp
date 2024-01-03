#include "utils.h"

#include <iostream>

#include "lapjv.h"

double lapjv(CostMatrix &cost, std::vector<int> &rowsol,
             std::vector<int> &colsol, bool extend_cost, float cost_limit,
             bool return_cost)
{
    std::vector<std::vector<float>> cost_c;

    for (Eigen::Index i = 0; i < cost.rows(); i++)
    {
        std::vector<float> row;
        for (Eigen::Index j = 0; j < cost.cols(); j++)
        {
            row.push_back(cost(i, j));
        }
        cost_c.push_back(row);
    }

    std::vector<std::vector<float>> cost_c_extended;

    int n_rows = static_cast<int>(cost.rows());
    int n_cols = static_cast<int>(cost.cols());
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols) { n = n_rows; }
    else
    {
        if (!extend_cost)
        {
            std::cout << "set extend_cost=True" << std::endl;
            exit(0);
        }
    }

    if (extend_cost || cost_limit < LONG_MAX)
    {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (int i = 0; i < cost_c_extended.size(); i++)
            cost_c_extended[i].resize(n);

        if (cost_limit < LONG_MAX)
        {
            for (int i = 0; i < cost_c_extended.size(); i++)
            {
                for (int j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_limit / 2.0;
                }
            }
        }
        else
        {
            float cost_max = -1;
            for (int i = 0; i < cost_c.size(); i++)
            {
                for (int j = 0; j < cost_c[i].size(); j++)
                {
                    if (cost_c[i][j] > cost_max) cost_max = cost_c[i][j];
                }
            }
            for (int i = 0; i < cost_c_extended.size(); i++)
            {
                for (int j = 0; j < cost_c_extended[i].size(); j++)
                {
                    cost_c_extended[i][j] = cost_max + 1;
                }
            }
        }

        for (int i = n_rows; i < cost_c_extended.size(); i++)
        {
            for (int j = n_cols; j < cost_c_extended[i].size(); j++)
            {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    double **cost_ptr;
    cost_ptr = new double *[n];
    for (int i = 0; i < n; i++) cost_ptr[i] = new double[n];

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++) { cost_ptr[i][j] = cost_c[i][j]; }
    }

    int *x_c = new int[n];
    int *y_c = new int[n];

    int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
    if (ret != 0)
    {
        std::cout << "Calculate Wrong!" << std::endl;
        exit(0);
    }

    double opt = 0.0;

    if (n != n_rows)
    {
        for (int i = 0; i < n; i++)
        {
            if (x_c[i] >= n_cols) x_c[i] = -1;
            if (y_c[i] >= n_rows) y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; i++) { rowsol[i] = x_c[i]; }
        for (int i = 0; i < n_cols; i++) { colsol[i] = y_c[i]; }

        if (return_cost)
        {
            for (int i = 0; i < rowsol.size(); i++)
            {
                if (rowsol[i] != -1) { opt += cost_ptr[i][rowsol[i]]; }
            }
        }
    }
    else if (return_cost)
    {
        for (int i = 0; i < rowsol.size(); i++)
        {
            opt += cost_ptr[i][rowsol[i]];
        }
    }

    for (int i = 0; i < n; i++) { delete[] cost_ptr[i]; }
    delete[] cost_ptr;
    delete[] x_c;
    delete[] y_c;

    return opt;
}
