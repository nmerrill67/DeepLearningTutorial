DIR = build
CMD = pdflatex --output-directory=$(DIR) --halt-on-error
PAPER = main

all:
	mkdir -p $(DIR) && $(CMD) $(PAPER) && $(CMD) $(PAPER) && mv $(DIR)/$(PAPER).pdf $(PAPER).pdf

clean:
	rm -rf $(DIR)

view:
	evince $(PAPER).pdf &
	
