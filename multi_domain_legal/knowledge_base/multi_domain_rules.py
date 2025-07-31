"""
Multi-Domain Legal Knowledge Base.

This module contains Prolog rules for legal reasoning across all domains:
- Legal Aid and Access to Justice
- Family Law and Personal Status  
- Consumer Protection and Rights
- Fundamental Rights and Constitutional Law
- Employment Law and Labor Rights
"""

MULTI_DOMAIN_LEGAL_RULES = """
% ============================================================================
% LEGAL AID DOMAIN RULES
% ============================================================================

% Income-based eligibility for legal aid
eligible_for_legal_aid(Person) :-
    person(Person),
    (   income_eligible(Person)
    ;   categorically_eligible(Person)
    ).

% Income eligibility rules
income_eligible(Person) :-
    annual_income(Person, Income),
    Income =< 300000.  % Rs. 3 lakh per annum

income_eligible(Person) :-
    monthly_income(Person, Income),
    Income =< 25000.   % Rs. 25,000 per month

% Categorical eligibility
categorically_eligible(Person) :-
    social_category(Person, Category),
    member(Category, [sc, st, obc]).

categorically_eligible(Person) :-
    vulnerable_group(Person, Group),
    member(Group, [women, children, disabled, industrial_workers]).

% Case type eligibility
legal_aid_applicable(Person, CaseType) :-
    eligible_for_legal_aid(Person),
    covered_case_type(CaseType).

covered_case_type(CaseType) :-
    member(CaseType, [criminal, family, labor, consumer, constitutional]).

excluded_case_type(CaseType) :-
    member(CaseType, [defamation, corporate_disputes, tax_evasion]).

% ============================================================================
% FAMILY LAW DOMAIN RULES  
% ============================================================================

% Marriage validity rules
valid_marriage(Person1, Person2, Law) :-
    personal_law(Person1, Law),
    personal_law(Person2, Law),
    marriage_conditions_met(Person1, Person2, Law).

% Age requirements for marriage
marriage_conditions_met(Person1, Person2, Law) :-
    age_requirement_met(Person1, Law),
    age_requirement_met(Person2, Law),
    not prohibited_relationship(Person1, Person2).

age_requirement_met(Person, hindu_law) :-
    age(Person, Age),
    gender(Person, male),
    Age >= 21.

age_requirement_met(Person, hindu_law) :-
    age(Person, Age),
    gender(Person, female),
    Age >= 18.

age_requirement_met(Person, muslim_law) :-
    age(Person, Age),
    Age >= 15.  % Puberty age

age_requirement_met(Person, christian_law) :-
    age(Person, Age),
    Age >= 21.

% Divorce grounds
divorce_grounds_exist(Person, Spouse, Grounds) :-
    personal_law(Person, Law),
    valid_divorce_ground(Grounds, Law).

valid_divorce_ground(adultery, hindu_law).
valid_divorce_ground(cruelty, hindu_law).
valid_divorce_ground(desertion, hindu_law).
valid_divorce_ground(conversion, hindu_law).
valid_divorce_ground(mental_disorder, hindu_law).

valid_divorce_ground(khula, muslim_law).
valid_divorce_ground(mubarat, muslim_law).
valid_divorce_ground(talaq, muslim_law).

% Maintenance eligibility
maintenance_eligible(Person) :-
    (   divorced_wife(Person)
    ;   separated_wife(Person)
    ;   child_of_divorced_parents(Person)
    ;   elderly_parent(Person)
    ).

maintenance_amount(Person, Amount) :-
    maintenance_eligible(Person),
    spouse_income(Person, SpouseIncome),
    Amount is SpouseIncome * 0.25.  % 25% of spouse's income

% Child custody rules
child_custody_preference(Child, Parent) :-
    age(Child, Age),
    Age < 5,
    gender(Parent, female).  % Tender years doctrine

child_custody_preference(Child, Parent) :-
    age(Child, Age),
    Age >= 5,
    best_interest_of_child(Child, Parent).

% ============================================================================
% CONSUMER PROTECTION DOMAIN RULES
% ============================================================================

% Forum jurisdiction rules
consumer_forum_jurisdiction(Amount, Forum) :-
    Amount =< 2000000,
    Forum = district_forum.

consumer_forum_jurisdiction(Amount, Forum) :-
    Amount > 2000000,
    Amount =< 10000000,
    Forum = state_commission.

consumer_forum_jurisdiction(Amount, Forum) :-
    Amount > 10000000,
    Forum = national_commission.

% Consumer complaint validity
valid_consumer_complaint(Person, Issue) :-
    consumer(Person),
    consumer_issue(Issue).

consumer_issue(defective_goods).
consumer_issue(deficient_services).
consumer_issue(unfair_trade_practice).
consumer_issue(excessive_price).
consumer_issue(false_advertisement).

% Complaint time limit
complaint_within_time_limit(ComplaintDate, TransactionDate) :-
    days_between(TransactionDate, ComplaintDate, Days),
    Days =< 730.  % 2 years from transaction

% Compensation calculation
consumer_compensation(Person, TransactionAmount, Compensation) :-
    valid_consumer_complaint(Person, _),
    defective_goods_compensation(TransactionAmount, Compensation).

defective_goods_compensation(Amount, Compensation) :-
    Compensation is Amount + (Amount * 0.12).  % Refund + 12% interest

% ============================================================================
% FUNDAMENTAL RIGHTS DOMAIN RULES
% ============================================================================

% Constitutional rights violations
fundamental_right_violated(Person, Right) :-
    citizen(Person),
    constitutional_right(Right),
    right_violation_occurred(Person, Right).

constitutional_right(right_to_equality).
constitutional_right(right_to_freedom).
constitutional_right(right_against_exploitation).
constitutional_right(right_to_freedom_of_religion).
constitutional_right(cultural_and_educational_rights).
constitutional_right(right_to_constitutional_remedies).

% Article 14 - Right to Equality
right_violation_occurred(Person, right_to_equality) :-
    discriminated_against(Person, Grounds),
    prohibited_discrimination_ground(Grounds).

prohibited_discrimination_ground(religion).
prohibited_discrimination_ground(race).
prohibited_discrimination_ground(caste).
prohibited_discrimination_ground(sex).
prohibited_discrimination_ground(place_of_birth).

% Article 19 - Right to Freedom
right_violation_occurred(Person, right_to_freedom) :-
    (   freedom_of_speech_violated(Person)
    ;   freedom_of_assembly_violated(Person)
    ;   freedom_of_movement_violated(Person)
    ;   freedom_of_profession_violated(Person)
    ).

% RTI eligibility
rti_applicable(Person, Information) :-
    citizen(Person),
    public_information(Information),
    not exempt_information(Information).

exempt_information(national_security).
exempt_information(cabinet_papers).
exempt_information(personal_information).

% PIL standing
pil_standing(Person, Issue) :-
    (   affected_by_issue(Person, Issue)
    ;   public_interest_issue(Issue)
    ).

public_interest_issue(environmental_violation).
public_interest_issue(corruption_in_government).
public_interest_issue(violation_of_fundamental_rights).

% ============================================================================
% EMPLOYMENT LAW DOMAIN RULES
% ============================================================================

% Wrongful termination rules
wrongful_termination(Employee) :-
    terminated(Employee),
    (   no_valid_reason(Employee)
    ;   improper_procedure(Employee)
    ;   discriminatory_termination(Employee)
    ).

improper_procedure(Employee) :-
    terminated(Employee),
    (   no_show_cause_notice(Employee)
    ;   no_domestic_inquiry(Employee)
    ;   insufficient_notice_period(Employee)
    ).

% Notice period requirements
sufficient_notice_period(Employee, Days) :-
    employment_type(Employee, permanent),
    Days >= 30.

sufficient_notice_period(Employee, Days) :-
    employment_type(Employee, temporary),
    Days >= 14.

% Retrenchment compensation
retrenchment_compensation(Employee, Amount) :-
    retrenched(Employee),
    service_years(Employee, Years),
    last_drawn_salary(Employee, Salary),
    Amount is Years * Salary * 15 / 26.  % 15 days wage per year

% Sexual harassment complaint validity
valid_harassment_complaint(Employee) :-
    sexual_harassment_occurred(Employee),
    complaint_within_time_limit(Employee, 90).  % 3 months

harassment_remedy_available(Employee, Remedy) :-
    valid_harassment_complaint(Employee),
    harassment_remedy(Remedy).

harassment_remedy(written_apology).
harassment_remedy(warning_to_harasser).
harassment_remedy(transfer_of_harasser).
harassment_remedy(compensation_to_victim).

% Minimum wage compliance
minimum_wage_violation(Employee) :-
    employee(Employee),
    wage_category(Employee, Category),
    actual_wage(Employee, ActualWage),
    minimum_wage_rate(Category, MinWage),
    ActualWage < MinWage.

minimum_wage_rate(unskilled, 178).    % per day
minimum_wage_rate(semi_skilled, 196).
minimum_wage_rate(skilled, 215).
minimum_wage_rate(highly_skilled, 236).

% Overtime payment
overtime_payment_due(Employee, Amount) :-
    daily_hours_worked(Employee, Hours),
    Hours > 8,
    overtime_hours(Employee, OvertimeHours),
    daily_wage(Employee, DailyWage),
    HourlyWage is DailyWage / 8,
    Amount is OvertimeHours * HourlyWage * 2.  % Double rate

overtime_hours(Employee, OvertimeHours) :-
    daily_hours_worked(Employee, Hours),
    Hours > 8,
    OvertimeHours is Hours - 8.

% ============================================================================
% CROSS-DOMAIN RULES
% ============================================================================

% Legal aid for employment matters
legal_aid_employment_case(Employee) :-
    eligible_for_legal_aid(Employee),
    (   wrongful_termination(Employee)
    ;   minimum_wage_violation(Employee)
    ;   valid_harassment_complaint(Employee)
    ).

% Legal aid for family matters
legal_aid_family_case(Person) :-
    eligible_for_legal_aid(Person),
    (   maintenance_eligible(Person)
    ;   divorce_case(Person)
    ;   child_custody_case(Person)
    ).

% Legal aid for consumer matters
legal_aid_consumer_case(Person) :-
    eligible_for_legal_aid(Person),
    valid_consumer_complaint(Person, _),
    transaction_amount(Person, Amount),
    Amount =< 100000.  % Small value cases

% Constitutional remedy for employment discrimination
constitutional_employment_remedy(Employee) :-
    employee(Employee),
    discriminated_against(Employee, Grounds),
    prohibited_discrimination_ground(Grounds),
    workplace_discrimination(Employee).

% PIL for employment issues
employment_pil_standing(Person, Issue) :-
    public_interest_issue(Issue),
    employment_related_issue(Issue).

employment_related_issue(widespread_minimum_wage_violation).
employment_related_issue(systemic_workplace_harassment).
employment_related_issue(mass_illegal_terminations).

% ============================================================================
% UTILITY RULES
% ============================================================================

% Date calculations
days_between(Date1, Date2, Days) :-
    % Simplified - in practice would use proper date arithmetic
    Days is 30.  % Placeholder

% Court hierarchy and appeals
appeal_court(district_court, high_court).
appeal_court(high_court, supreme_court).
appeal_court(consumer_district_forum, state_consumer_commission).
appeal_court(state_consumer_commission, national_consumer_commission).

% Legal costs estimation
legal_costs_estimate(CaseType, Amount, Costs) :-
    base_court_fee(CaseType, BaseFee),
    lawyer_fee_estimate(Amount, LawyerFee),
    Costs is BaseFee + LawyerFee.

base_court_fee(civil, 500).
base_court_fee(criminal, 100).
base_court_fee(family, 300).
base_court_fee(consumer, 200).

lawyer_fee_estimate(Amount, Fee) :-
    Amount =< 100000,
    Fee = 10000.

lawyer_fee_estimate(Amount, Fee) :-
    Amount > 100000,
    Amount =< 1000000,
    Fee is Amount * 0.05.

% Document requirements
required_documents(legal_aid, [income_certificate, caste_certificate, case_documents]).
required_documents(marriage, [age_proof, address_proof, photographs]).
required_documents(divorce, [marriage_certificate, evidence_of_grounds]).
required_documents(consumer_complaint, [purchase_receipt, communication_records]).
required_documents(employment_case, [appointment_letter, salary_slips, termination_notice]).

% Timeline estimates
case_timeline_estimate(legal_aid_application, '15-30 days').
case_timeline_estimate(divorce_mutual_consent, '6-18 months').
case_timeline_estimate(divorce_contested, '2-5 years').
case_timeline_estimate(consumer_complaint, '3-12 months').
case_timeline_estimate(employment_dispute, '6 months - 2 years').
case_timeline_estimate(fundamental_rights_petition, '1-3 years').
"""
