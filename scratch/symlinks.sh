#!/bin/bash

SRC_BASE="/data/nas/motic"
TARGET="/data/mtanzer/autoscope_images"
SPLITS=(1 2 3)
DRY_RUN=false

if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] No changes will be made"
fi

mkdir -p "$TARGET"

for split in "${SPLITS[@]}"; do
    SRC="${SRC_BASE}/AutoscopeImagesP3_prefilter30_split${split}"

    if [ ! -d "$SRC" ]; then
        echo "WARNING: $SRC does not exist, skipping"
        continue
    fi

    echo "Processing split${split}..."

    for dirpath in "$SRC"/*/; do
        dirname=$(basename "$dirpath")
        target_link="$TARGET/$dirname"

        if [ -L "$target_link" ]; then
            echo "SKIP (already linked): $dirname"
        elif [ -e "$target_link" ]; then
            echo "COLLISION (non-symlink exists): $dirname"
        else
            if $DRY_RUN; then
                echo "LINK: $dirpath -> $target_link"
            else
                echo "LINK: $dirpath -> $target_link"
                ln -s "$dirpath" "$target_link"
            fi
        fi
    done
done

echo "Done. Total symlinks: $(find $TARGET -maxdepth 1 -type l | wc -l)"