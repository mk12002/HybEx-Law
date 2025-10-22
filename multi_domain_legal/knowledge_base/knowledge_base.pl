% =======================================================
% HYBEX-LAW MASTER KNOWLEDGE BASE
% This is the single entry point for the entire system.
% =======================================================

% Load foundational predicates first
:- include('foundational_rules_clean.pl').

% Load the authoritative eligibility rules
:- include('legal_aid_clean_v2.pl').

% Load domain-specific definitions
:- include('employment_law.pl').
:- include('family_law.pl').
:- include('consumer_protection.pl').

% Load rules that connect the domains
:- include('cross_domain_rules.pl').

% Load reasoning helpers for detailed explanations
:- include('reasoning_helpers.pl').
