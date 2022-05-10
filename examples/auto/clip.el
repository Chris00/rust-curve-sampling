(TeX-add-style-hook
 "clip"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt" "a4paper")))
   (TeX-run-style-hooks
    "latex2e"
    "clip1"
    "clip0"
    "article"
    "art12"
    "tikz"))
 :latex)

