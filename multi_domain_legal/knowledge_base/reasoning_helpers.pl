% ============================================================================
% FIXED reasoning_helpers.pl - v4.0
% USES CORRECT FACT STRUCTURE (e.g., annual_income/2, social_category/2)
% ============================================================================

% ============================================================================
% generate_detailed_reasoning/2 - WITH PROPER FALLBACKS
% ============================================================================

generate_detailed_reasoning(CaseID, DetailedReason) :-
    % Try to find eligibility factors
    (   findall(Factor, check_direct_eligibility_factor(CaseID, Factor), EligFactors),
        EligFactors \= []
    ->  % Success: found eligibility factors
        length(EligFactors, Count),
        atomic_list_concat(EligFactors, '; ', FactorsStr),
        atomic_list_concat(['Case ', CaseID, ' is ELIGIBLE based on ', Count, ' factor(s): ', FactorsStr], DetailedReason)
    ;   % No eligibility factors: check ineligibility
        findall(InFactor, check_direct_ineligibility_factor(CaseID, InFactor), IneligFactors),
        IneligFactors \= []
    ->  % Found ineligibility factors
        length(IneligFactors, InCount),
        atomic_list_concat(IneligFactors, '; ', InFactorsStr),
        atomic_list_concat(['Case ', CaseID, ' is NOT ELIGIBLE. ', InCount, ' reason(s): ', InFactorsStr], DetailedReason)
    ;   % Fallback: no factors found at all
        atomic_list_concat(['Case ', CaseID, ' - insufficient data for detailed analysis'], DetailedReason)
    ).

% ============================================================================
% check_direct_eligibility_factor/2 - USES CORRECT FACTS
% ============================================================================

% 1. NO INCOME
check_direct_eligibility_factor(CaseID, 'No income reported') :-
    no_income(CaseID).

% 2. VULNERABLE GROUPS (ANY)
check_direct_eligibility_factor(CaseID, 'Member of vulnerable group') :-
    vulnerable_group(CaseID, _).

% 3. LOW INCOME (example, adjust as needed)
check_direct_eligibility_factor(CaseID, 'Low income') :-
    annual_income(CaseID, Income),
    number(Income),
    Income =< 300000.

% 4. VULNERABLE CASTE
check_direct_eligibility_factor(CaseID, 'Vulnerable social category (SC/ST/BPL)') :-
    social_category(CaseID, Category),
    member(Category, ['sc', 'st', 'bpl']).

% 5. WRONGFUL TERMINATION
check_direct_eligibility_factor(CaseID, 'Wrongful termination case') :-
    wrongful_termination(CaseID).

% 6. DOMESTIC VIOLENCE
check_direct_eligibility_factor(CaseID, 'Domestic violence victim') :-
    case_type(CaseID, domestic_violence).
    
% 7. CONSUMER COMPLAINT
check_direct_eligibility_factor(CaseID, 'Valid consumer complaint') :-
    valid_consumer_complaint(CaseID, _).

% ============================================================================
% check_direct_ineligibility_factor/2 - USES CORRECT FACTS
% ============================================================================

% 1. HIGH INCOME (General)
check_direct_ineligibility_factor(CaseID, 'High income (General > 300k)') :-
    annual_income(CaseID, Income),
    number(Income),
    Income > 300000,
    social_category(CaseID, 'general'),
    \+ vulnerable_group(CaseID, _).

% 2. HIGH INCOME (SC/ST)
check_direct_ineligibility_factor(CaseID, 'High income (SC/ST > 800k)') :-
    annual_income(CaseID, Income),
    number(Income),
    Income > 800000,
    social_category(CaseID, Category),
    member(Category, ['sc', 'st']),
    \+ vulnerable_group(CaseID, _).

% ============================================================================
% primary_eligibility_reason/2 - PRIORITIZED DIRECT CHECKS
% ============================================================================

primary_eligibility_reason(CaseID, PrimaryReason) :-
    (   find_primary_eligible_reason(CaseID, PrimaryReason)
    ->  true
    ;   find_primary_ineligible_reason(CaseID, PrimaryReason)
    ->  true
    ;   PrimaryReason = 'Standard legal aid evaluation criteria'
    ).

% ELIGIBLE REASONS (PRIORITIZED)

find_primary_eligible_reason(CaseID, 'No income reported') :-
    no_income(CaseID), !.

find_primary_eligible_reason(CaseID, 'Member of vulnerable group') :-
    vulnerable_group(CaseID, _), !.

find_primary_eligible_reason(CaseID, 'Vulnerable caste (SC/ST/BPL)') :-
    social_category(CaseID, Category),
    member(Category, ['sc', 'st', 'bpl']), !.

find_primary_eligible_reason(CaseID, 'Low income') :-
    annual_income(CaseID, Income),
    number(Income),
    Income =< 300000, !.

find_primary_eligible_reason(_, 'General eligibility criteria met').


% INELIGIBLE REASONS (PRIORITIZED)

find_primary_ineligible_reason(CaseID, 'High income (General > 300k)') :-
    annual_income(CaseID, Income),
    number(Income),
    Income > 300000,
    social_category(CaseID, 'general'),
    \+ vulnerable_group(CaseID, _), !.
    
find_primary_ineligible_reason(CaseID, 'High income (SC/ST > 800k)') :-
    annual_income(CaseID, Income),
    number(Income),
    Income > 800000,
    social_category(CaseID, Category),
    member(Category, ['sc', 'st']),
    \+ vulnerable_group(CaseID, _), !.

find_primary_ineligible_reason(_, 'Does not meet minimum legal aid criteria').

% ============================================================================
% HELPER PREDICATES
% ============================================================================

count_eligibility_factors(CaseID, Count) :-
    findall(1, check_direct_eligibility_factor(CaseID, _), Ones),
    length(Ones, Count).

list_eligibility_factors(CaseID, FactorsList) :-
    findall(Factor, check_direct_eligibility_factor(CaseID, Factor), FactorsList).

% ============================================================================
% END v4.0
% ============================================================================
