#!/bin/bash
#
# Run checks related to code quality.
#
# This script is intended for both the CI and to check locally that code standards are
# respected. We are currently linting (PEP-8 and similar), looking for patterns of
# common mistakes (sphinx directives with missing blank lines, old style classes,
# unwanted imports...), we run doctests here (currently some files only), and we
# validate formatting error in docstrings.
#
# Usage:
#   $ ./ci/code_checks.sh               # run all checks
#   $ ./ci/code_checks.sh lint          # run linting only
#   $ ./ci/code_checks.sh patterns      # check for patterns that should not exist
#   $ ./ci/code_checks.sh code          # checks on imported code
#   $ ./ci/code_checks.sh doctests      # run doctests
#   $ ./ci/code_checks.sh docstrings    # validate docstring errors
#   $ ./ci/code_checks.sh dependencies  # check that dependencies are consistent
#   $ ./ci/code_checks.sh typing	# run static type analysis

[[ -z "$1" || "$1" == "lint" || "$1" == "patterns" || "$1" == "doctests" || "$1" == "docstrings" || "$1" == "dependencies" || "$1" == "typing" ]] || \
    { echo "Unknown command $1. Usage: $0 [lint|patterns|doctests|docstrings|dependencies|typing]"; exit 9999; }

BASE_DIR="$(dirname $0)/.."
RET=0
CHECK=$1

function invgrep {
    # grep with inverse exist status and formatting for azure-pipelines
    #
    # This function works exactly as grep, but with opposite exit status:
    # - 0 (success) when no patterns are found
    # - 1 (fail) when the patterns are found
    #
    # This is useful for the CI, as we want to fail if one of the patterns
    # that we want to avoid is found by grep.
    grep -n "$@" | sed "s/^/$INVGREP_PREPEND/" | sed "s/$/$INVGREP_APPEND/" ; EXIT_STATUS=${PIPESTATUS[0]}
    return $((! $EXIT_STATUS))
}

if [[ "$GITHUB_ACTIONS" == "true" ]]; then
    FLAKE8_FORMAT="##[error]%(path)s:%(row)s:%(col)s:%(code)s:%(text)s"
    INVGREP_PREPEND="##[error]"
else
    FLAKE8_FORMAT="default"
fi

### LINTING ###
if [[ -z "$CHECK" || "$CHECK" == "lint" ]]; then

    echo "black --version"
    black --version

    MSG='Checking black formatting' ; echo $MSG
    black . --check
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    # `setup.cfg` contains the list of error codes that are being ignored in flake8
    echo "flake8 --version"
    flake8 --version

    # hydrodata/_libs/src is C code, so no need to search there.
    MSG='Linting .py code' ; echo $MSG
    flake8 --format="$FLAKE8_FORMAT" .
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Linting code-blocks in .rst documentation' ; echo $MSG
    flake8 --select RST docs --filename=*.rst --format="$FLAKE8_FORMAT"
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of not concatenated strings' ; echo $MSG
    if [[ "$GITHUB_ACTIONS" == "true" ]]; then
        $BASE_DIR/scripts/validate_unwanted_patterns.py --validation-type="strings_to_concatenate" --format="##[error]{source_path}:{line_number}:{msg}" .
    else
        $BASE_DIR/scripts/validate_unwanted_patterns.py --validation-type="strings_to_concatenate" .
    fi
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for strings with wrong placed spaces' ; echo $MSG
    if [[ "$GITHUB_ACTIONS" == "true" ]]; then
        $BASE_DIR/scripts/validate_unwanted_patterns.py --validation-type="strings_with_wrong_placed_whitespace" --format="##[error]{source_path}:{line_number}:{msg}" .
    else
        $BASE_DIR/scripts/validate_unwanted_patterns.py --validation-type="strings_with_wrong_placed_whitespace" .
    fi
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    echo "isort --version-number"
    isort --version-number

    # Imports - Check formatting using isort see setup.cfg for settings
    MSG='Check import format using isort' ; echo $MSG
    ISORT_CMD="isort --quiet --recursive --check-only hydrodata scripts"
    if [[ "$GITHUB_ACTIONS" == "true" ]]; then
        eval $ISORT_CMD | awk '{print "##[error]" $0}'; RET=$(($RET + ${PIPESTATUS[0]}))
    else
        eval $ISORT_CMD
    fi
    RET=$(($RET + $?)) ; echo $MSG "DONE"

fi

### PATTERNS ###
if [[ -z "$CHECK" || "$CHECK" == "patterns" ]]; then

    MSG='Check for python2-style super usage' ; echo $MSG
    invgrep -R --include="*.py" -E "super\(\w*, (self|cls)\)" hydrodata
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for deprecated messages without sphinx directive' ; echo $MSG
    invgrep -R --include="*.py" --include="*.pyx" -E "(DEPRECATED|DEPRECATE|Deprecated)(:|,|\.)" hydrodata
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for backticks incorrectly rendering because of missing spaces' ; echo $MSG
    invgrep -R --include="*.rst" -E "[a-zA-Z0-9]\`\`?[a-zA-Z0-9]" docs
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for incorrect sphinx directives' ; echo $MSG
    invgrep -R --include="*.py" --include="*.pyx" --include="*.rst" -E "\.\. (autosummary|contents|currentmodule|deprecated|function|image|important|include|ipython|literalinclude|math|module|note|raw|seealso|toctree|versionadded|versionchanged|warning):[^:]" ./hydrodata ./docs
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for wrong space after ipython directive and before colon (".. ipython ::" instead of ".. ipython::")' ; echo $MSG
    invgrep -R --include="*.rst" ".. ipython ::" docs
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for extra blank lines after the class definition' ; echo $MSG
    invgrep -R --include="*.py" --include="*.pyx" -E 'class.*:\n\n( )+"""' .
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of {foo!r} instead of {repr(foo)}' ; echo $MSG
    invgrep -R --include=*.{py,pyx} '!r}' hydrodata
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of comment-based annotation syntax' ; echo $MSG
    invgrep -R --include="*.py" -P '# type: (?!ignore)' hydrodata
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of foo.__class__ instead of type(foo)' ; echo $MSG
    invgrep -R --include=*.{py,pyx} '\.__class__' hydrodata
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check for use of xrange instead of range' ; echo $MSG
    invgrep -R --include=*.{py,pyx} 'xrange' hydrodata
    RET=$(($RET + $?)) ; echo $MSG "DONE"

    MSG='Check that no file in the repo contains trailing whitespaces' ; echo $MSG
    INVGREP_APPEND=" <- trailing whitespaces found"
    invgrep -RI --exclude=\*.{svg,c,cpp,html,js} --exclude-dir=env "\s$" *
    RET=$(($RET + $?)) ; echo $MSG "DONE"
    unset INVGREP_APPEND
fi

### DOCTESTS ###
if [[ -z "$CHECK" || "$CHECK" == "doctests" ]]; then

    MSG='Doctests' ; echo $MSG
    pytest -q --doctest-modules hydrodata
    RET=$(($RET + $?)) ; echo $MSG "DONE"

fi

### DOCSTRINGS ###
if [[ -z "$CHECK" || "$CHECK" == "docstrings" ]]; then

    MSG='Validate docstrings (GL03, GL04, GL05, GL06, GL07, GL09, GL10, SS04, SS05, PR03, PR04, PR05, PR10, EX04, RT01, RT04, RT05, SA02, SA03)' ; echo $MSG
    $BASE_DIR/scripts/validate_docstrings.py --format=actions --errors=GL03,GL04,GL05,GL06,GL07,GL09,GL10,SS04,SS05,PR03,PR04,PR05,PR10,EX04,RT01,RT04,RT05,SA02,SA03
    RET=$(($RET + $?)) ; echo $MSG "DONE"

fi

### DEPENDENCIES ###
if [[ -z "$CHECK" || "$CHECK" == "dependencies" ]]; then

    MSG='Check that requirements-dev.txt has been generated from environment.yml' ; echo $MSG
    $BASE_DIR/scripts/generate_pip_deps_from_conda.py --compare
    RET=$(($RET + $?)) ; echo $MSG "DONE"

fi

### TYPING ###
# if [[ -z "$CHECK" || "$CHECK" == "typing" ]]; then

#     echo "mypy --version"
#     mypy --version

#     MSG='Performing static analysis using mypy' ; echo $MSG
#     mypy hydrodata
#     RET=$(($RET + $?)) ; echo $MSG "DONE"
# fi


exit $RET
