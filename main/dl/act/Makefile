python = python3.8
srcs = $(wildcard ./_*.py)

test:
	@for src in $(srcs); do \
		$(python) $$src; \
	done

imgs = $(wildcard *.png)
img_path = ./images
move:
	@mkdir -p $(img_path)
	@for img in $(imgs); do \
		mv $$img images/$$img; \
	done
