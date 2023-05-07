#!/opt/homebrew/bin/bash -e

function make {
    local file="$1"
    
    pdflatex $file.tex
    bibtex   $file.aux
    pdflatex $file.tex
    pdflatex $file.tex
}


function make_section {
    local section="$1"
    
    file=$(find ./content -name $section.tex)
    if [[ -z $file ]]; then
        echo "No such section"
        exit 1
    fi

    pushd $(dirname "$file")
        make $section
    popd
}


function clean {
    for ext in aux bbl blg fdb_latexmk fls log nav out snm toc synctex.gz; do
        find . -name "*.${ext}" -type f -delete
    done
}


if [[ -z "$1" ]]; then
    make doc
    exit
fi

if [[ "$1" == "clean" ]]; then
    clean
    exit
fi

# Make section separate
make_section "$1"
