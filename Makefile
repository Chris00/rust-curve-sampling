

clip:
	cargo run --example clip
	cp examples/clip.tex /tmp
	cd /tmp && pdflatex clip


.PHONY: clip
