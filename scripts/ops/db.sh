#!/bin/zsh
## Create the wikipedia sqlite db.
## Create the tables.
## Populate the tables.
## Index the title and sentence index columns.
## Printout the stats.
## Note: No preprocessing is done here.

DB="wikipedia.sqlite"
TABLE="Wikipedia"
DIR_DATASET="../../dataset/wiki-pages-text"
#DIR_DATASET="."
echo "DB=${DB}"
echo "TABLE=${TABLE}"
echo "DIR_DATASET=${DIR_DATASET}"

# 1. check dependencies before proceeding.
[[ -f 'refresh.sql' ]] || ( >&2 echo "-- Missing refresh.sql" && exit 1 )
[[ -f 'format.py' ]] || ( >&2 echo "-- Missing format.py" && exit 1 )

# 2. build the database
echo "Rebuild database? (y/n)"
read x;
if [[ x == 'y' ]]; then
  echo "++ Building the database..."
  sqlite3 ${DB} < <(sed 's/TABLE_PLACEHOLDER/'"${TABLE}"'/g' refresh.sql)
fi;

sqlite3 ${DB} <<< "PRAGMA table_info('${TABLE}');"
sqlite3 ${DB} <<< ".indexes"

# 3. Populate the database
wiki_files=( "$DIR_DATASET"/*.txt )

time (
  for file in ${wiki_files}; do
    echo "++ Processing $file..."
    tf=$(mktemp --suffix .processed)
    python format.py "${file}" > "${tf}"
    sqlite3 ${DB} <<< ".import ${tf} ${TABLE}";
    if [[ $? -eq 0 ]]; then
      rm -f "$tf";
    else
      echo "-- Failed to import $tf... this file is not removed."
    fi
  done
)
echo "++ All wiki files processed."

# 4. print stats
sqlite3 ${DB} <<< "SELECT * FROM ${TABLE} LIMIT 10;"
sqlite3 ${DB} <<< "SELECT 'Count ' || count(*) FROM ${TABLE};"
du -h ${DB}