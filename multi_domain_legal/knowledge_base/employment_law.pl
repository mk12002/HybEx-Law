<<<<<<< HEAD

=======
% EM% =================================================================
% EMPLOYMENT MISCONDUCT AND TERMINATION RULES
% =================================================================

wrongful_termination(Employee) :-
    improper_procedure(Employee).
wrongful_termination(Employee) :-
    insufficient_notice(Employee).
wrongful_termination(Employee) :-
    discriminatory_termination(Employee).

% Improved procedure checks - avoid problematic negations
improper_procedure(Employee) :-
    disciplinary_hearing_conducted(Employee, false).
improper_procedure(Employee) :-
    opportunity_to_explain_given(Employee, false).MAIN RULES - CORRECTED
% Contains wrongful termination, wage disputes, and workplace harassment rules
% Corrected to avoid problematic cuts that prevent backtracking
>>>>>>> f63cb0c5bec52c3c68eb36a972ccaa75026c0afe

% =================================================================
% WRONGFUL TERMINATION - CORRECTED
% =================================================================

wrongful_termination(Employee) :-
    improper_procedure(Employee).
wrongful_termination(Employee) :-
    insufficient_notice(Employee).
wrongful_termination(Employee) :-
    discriminatory_termination(Employee).

<<<<<<< HEAD
% Add this new definition
discriminatory_termination(Employee) :-
    constitutional_employment_remedy(non_discrimination),
    case_type(Employee, discrimination).
    
=======
>>>>>>> f63cb0c5bec52c3c68eb36a972ccaa75026c0afe
improper_procedure(Employee) :-
    \+ disciplinary_hearing_conducted(Employee).
improper_procedure(Employee) :-
    \+ opportunity_to_explain_given(Employee).

insufficient_notice(Employee) :-
    notice_period_given(Employee, GivenPeriod),
    required_notice_period(Employee, RequiredPeriod),
    integer(GivenPeriod),
    integer(RequiredPeriod),
    GivenPeriod < RequiredPeriod.

sufficient_notice_period(Employee, GivenPeriod) :-
    required_notice_period(Employee, RequiredPeriod),
    integer(GivenPeriod),
    integer(RequiredPeriod),
    GivenPeriod >= RequiredPeriod.

required_notice_period(Employee, 30) :-
    employment_duration(Employee, Duration),
    integer(Duration),
    Duration < 365.  % Less than 1 year - 30 days
required_notice_period(Employee, 60) :-
    employment_duration(Employee, Duration),
    integer(Duration),
    Duration >= 365.  % 1 year or more - 60 days

% =================================================================
% RETRENCHMENT COMPENSATION
% =================================================================

retrenchment_compensation(Employee, Amount) :-
    employment_duration(Employee, Days),
    daily_wage(Employee, Wage),
    Amount is (Days / 365) * 15 * Wage, !.  % 15 days wage per year

% =================================================================
% WORKPLACE HARASSMENT
% =================================================================

valid_harassment_complaint(Employee) :-
    harassment_incident_reported(Employee),
    reported_within_time_limit(Employee), !.

harassment_remedy_available(Employee, Remedy) :-
    valid_harassment_complaint(Employee),
    harassment_remedy(Remedy), !.

harassment_remedy(compensation).
harassment_remedy(transfer).
harassment_remedy(disciplinary_action).
harassment_remedy(awareness_training).

% =================================================================
% MINIMUM WAGE VIOLATIONS
% =================================================================

minimum_wage_violation(Employee) :-
    actual_wage(Employee, ActualWage),
    minimum_wage_rate(Employee, MinWage),
    ActualWage < MinWage, !.

minimum_wage_rate(Employee, Rate) :-
    employment_state(Employee, State),
    state_minimum_wage(State, Rate), !.
minimum_wage_rate(_, 176) :- !.  % Default national minimum wage

overtime_payment_due(Employee, Amount) :-
    overtime_hours(Employee, Hours),
    Hours > 0,
    hourly_wage(Employee, Wage),
    Amount is Hours * Wage * 2, !.  % Double rate for overtime

overtime_hours(Employee, OvertimeHours) :-
    daily_working_hours(Employee, DailyHours),
    DailyHours > 8,
    OvertimeHours is DailyHours - 8, !.
overtime_hours(_, 0) :- !.  % No overtime

% =================================================================
% EMPLOYMENT LEGAL AID
% =================================================================

legal_aid_employment_case(Person) :-
    eligible_for_legal_aid(Person),  % From legal_aid.pl
    employment_case_type(Person, _), !.

employment_case_type(Person, wrongful_termination) :-
    case_type(Person, employment),
    wrongful_termination(Person), !.
employment_case_type(Person, wage_dispute) :-
    case_type(Person, wage_dispute), !.
employment_case_type(Person, harassment) :-
    case_type(Person, harassment), !.

% =================================================================
% CONSTITUTIONAL EMPLOYMENT REMEDIES
% =================================================================

constitutional_employment_remedy(equal_pay).
constitutional_employment_remedy(non_discrimination).
constitutional_employment_remedy(safe_working_conditions).

employment_pil_standing(Person, Issue) :-
    employment_related_issue(Issue),
    public_interest_affected(Issue), !.

<<<<<<< HEAD
% Define what constitutes a public interest issue for employment
public_interest_affected(Issue) :-
    employment_related_issue(Issue). % Assumes all defined employment issues affect public interest

=======
>>>>>>> f63cb0c5bec52c3c68eb36a972ccaa75026c0afe
employment_related_issue(bonded_labor).
employment_related_issue(child_labor).
employment_related_issue(unsafe_working_conditions).
employment_related_issue(gender_discrimination).
