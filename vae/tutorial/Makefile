DIR = build
CMD = pdflatex --output-directory=$(DIR) --halt-on-error
PAPER = main
PDF = MerrillVAE

all:
	mkdir -p $(DIR)
	$(CMD) $(PAPER) && biber $(DIR)/$(PAPER) && $(CMD) $(PAPER) && $(CMD) $(PAPER) && mv $(DIR)/$(PAPER).pdf $(PDF).pdf
clean:
	rm -rf $(DIR)
view:
	evince $(PDF) &
	
