doc:
	cargo doc --all-features

check:
	cargo semver-checks
	cargo deny check

clip:
	cargo run --example clip
	cp examples/clip.tex /tmp
	cd /tmp && pdflatex -interaction=batchmode clip

latex:
	cargo run --example latex_speed
	cd /tmp && time pdflatex -interaction=nonstopmode latex_speed > /dev/null

horror:
	@cargo test --lib -- --nocapture tests::horror && \
	cd target/ && gnuplot horror.gp && \
	echo "⇒ see target/horror.pdf"

miri:
	MIRIFLAGS="-Zmiri-disable-isolation" \
	cargo miri test


.PHONY: doc check clip latex horror miri
