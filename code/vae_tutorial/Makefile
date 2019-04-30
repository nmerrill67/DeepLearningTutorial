DIR = build
CMD = pdflatex --output-directory=$(DIR) --halt-on-error
PAPER = main
PDF = MerrillVAEpdf

all:
	mkdir -p $(DIR)
	$(CMD) $(PAPER) && biber $(DIR)/$(PAPER) && $(CMD) $(PAPER) && $(CMD) $(PAPER) && mv $(DIR)/$(PAPER).pdf $(PDF)
clean:
	rm -rf $(DIR)

view:
	evince $(PDF) &
	
