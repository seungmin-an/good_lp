//! A solver that uses [highs](https://docs.rs/highs), a parallel C++ solver.

use highs::HighsModelStatus;

use crate::solvers::{
    ObjectiveDirection, ResolutionError, Solution, SolutionWithDual, SolverModel,
};
use crate::{
    constraint::ConstraintReference,
    solvers::DualValues,
    variable::{UnsolvedProblem, VariableDefinition},
};
use crate::{Constraint, IntoAffineExpression, Variable};

/// The [highs](https://docs.rs/highs) solver,
/// to be used with [UnsolvedProblem::using].
///
/// This solver does not support integer variables and will panic
/// if given a problem with integer variables.
pub fn highs(to_solve: UnsolvedProblem) -> HighsProblem {
    let mut highs_problem = highs::RowProblem::default();
    let sense = match to_solve.direction {
        ObjectiveDirection::Maximisation => highs::Sense::Maximise,
        ObjectiveDirection::Minimisation => highs::Sense::Minimise,
    };
    let mut columns = Vec::with_capacity(to_solve.variables.len());
    for (
        var,
        &VariableDefinition {
            min,
            max,
            is_integer,
            ..
        },
    ) in to_solve.variables.iter_variables_with_def()
    {
        let &col_factor = to_solve
            .objective
            .linear
            .coefficients
            .get(&var)
            .unwrap_or(&0.);
        let col = highs_problem.add_column_with_integrality(col_factor, min..max, is_integer);
        columns.push(col);
    }
    HighsProblem {
        sense,
        highs_problem,
        columns,
        verbose: false,
        time_limit: None,
        time_limit_as_error: false,
        thread_nums: None,
        presolve: None,
    }
}

/// A HiGHS model
#[derive(Debug)]
pub struct HighsProblem {
    sense: highs::Sense,
    highs_problem: highs::RowProblem,
    columns: Vec<highs::Col>,
    verbose: bool,
    time_limit: Option<f64>,
    time_limit_as_error: bool,
    thread_nums: Option<i32>,
    presolve: Option<&'static str>,
}

impl HighsProblem {
    /// Get a highs model for this problem
    pub fn into_inner(self) -> highs::Model {
        self.highs_problem.optimise(self.sense)
    }

    /// Sets whether or not HiGHS should display verbose logging information to the console
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose
    }

    /// Sets time limit
    pub fn set_time_limit(&mut self, time_limit: f64) {
        self.time_limit = Some(time_limit);
    }

    /// Consider time limit as an error
    pub fn set_time_limit_as_error(&mut self, time_limit_as_error: bool) {
        self.time_limit_as_error = time_limit_as_error;
    }

    /// Sets the # of threads for solver
    pub fn set_thread_nums(&mut self, thread_nums: i32) {
        assert!(thread_nums >= 1);
        self.thread_nums = Some(thread_nums);
    }

    /// Sets the presolve
    pub fn set_presolve(&mut self, mode: &'static str) {
        assert!(mode == "on" || mode == "off" || mode == "choose");
        self.presolve = Some(mode);
    }
}

impl SolverModel for HighsProblem {
    type Solution = HighsSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let (verbose, time_limit, time_limit_as_error, thread_nums, presolve) = (
            self.verbose,
            self.time_limit,
            self.time_limit_as_error,
            self.thread_nums,
            self.presolve,
        );

        let mut model = self.into_inner();
        if verbose {
            model.set_option(&b"output_flag"[..], true);
            model.set_option(&b"log_to_console"[..], true);
            model.set_option(&b"log_dev_level"[..], 2);
        }
        if let Some(time_limit) = time_limit {
            model.set_option("time_limit", time_limit);
        }
        if let Some(thread_nums) = thread_nums {
            if thread_nums > 1 {
                model.set_option("parallel", "on");
                model.set_option("threads", thread_nums);
            }
        }
        if let Some(presolve) = presolve {
            model.set_option("presolve", presolve);
        }

        let solved = model.solve();
        match solved.status() {
            HighsModelStatus::NotSet => Err(ResolutionError::Other("NotSet")),
            HighsModelStatus::LoadError => Err(ResolutionError::Other("LoadError")),
            HighsModelStatus::ModelError => Err(ResolutionError::Other("ModelError")),
            HighsModelStatus::PresolveError => Err(ResolutionError::Other("PresolveError")),
            HighsModelStatus::SolveError => Err(ResolutionError::Other("SolveError")),
            HighsModelStatus::PostsolveError => Err(ResolutionError::Other("PostsolveError")),
            HighsModelStatus::ModelEmpty => Err(ResolutionError::Other("ModelEmpty")),
            HighsModelStatus::Infeasible => Err(ResolutionError::Infeasible),
            HighsModelStatus::Unbounded => Err(ResolutionError::Unbounded),
            HighsModelStatus::UnboundedOrInfeasible => Err(ResolutionError::Infeasible),
            HighsModelStatus::ReachedTimeLimit if time_limit_as_error => {
                Err(ResolutionError::Other("ReachedTimeLimit"))
            }
            HighsModelStatus::ReachedIterationLimit if time_limit_as_error => {
                Err(ResolutionError::Other("ReachedIterationLimit"))
            }
            HighsModelStatus::Unknown => Err(ResolutionError::Other("Unknown")),
            _ok_status => Ok(HighsSolution {
                solution: solved.get_solution(),
            }),
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.highs_problem.num_rows();
        let upper_bound = -constraint.expression.constant();
        let columns = &self.columns;
        let factors = constraint
            .expression
            .linear_coefficients()
            .into_iter()
            .map(|(variable, factor)| (columns[variable.index()], factor));
        if constraint.is_equality {
            self.highs_problem
                .add_row(upper_bound..=upper_bound, factors);
        } else {
            self.highs_problem.add_row(..=upper_bound, factors);
        }
        ConstraintReference { index }
    }
}

/// The solution to a highs problem
#[derive(Debug)]
pub struct HighsSolution {
    solution: highs::Solution,
}

impl HighsSolution {
    /// Returns the highs solution object. You can use it to fetch dual values
    pub fn into_inner(self) -> highs::Solution {
        self.solution
    }

    pub fn objective(&self) -> f64 {
        self.solution.objective()
    }
}

impl Solution for HighsSolution {
    fn value(&self, variable: Variable) -> f64 {
        self.solution.columns()[variable.index()]
    }
}

impl<'a> DualValues for &'a HighsSolution {
    fn dual(&self, constraint: ConstraintReference) -> f64 {
        self.solution.dual_rows()[constraint.index]
    }
}

impl<'a> SolutionWithDual<'a> for HighsSolution {
    type Dual = &'a HighsSolution;

    fn compute_dual(&'a mut self) -> &'a HighsSolution {
        self
    }
}
