

clip:
	cargo run --example clip
	cp examples/clip.tex /tmp
	cd /tmp && pdflatex -interaction=batchmode clip

miri:
	MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-tag-raw-pointers" \
	cargo miri test


.PHONY: clip
