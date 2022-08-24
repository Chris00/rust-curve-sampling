

clip:
	cargo run --example clip
	cp examples/clip.tex /tmp
	cd /tmp && pdflatex -interaction=batchmode clip

latex:
	cargo run --example latex_speed
	cd /tmp && time pdflatex -interaction=nonstopmode latex_speed > /dev/null

miri:
	MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-tag-raw-pointers" \
	cargo miri test


.PHONY: clip latex
