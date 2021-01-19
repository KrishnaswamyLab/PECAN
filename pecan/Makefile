annulus:
	@for n in 16 32 64 128 256 512; do \
		for r in 0.01 0.25 0.50 0.75 0.99; do \
			python condensation.py -r $$r -n $$n -d 'annulus' -o 'annulus_n'$$n'_r'$$r'.npz'; \
		done \
	done 

# 'Petals' data set to check what happens if we start with large-scale
# cycles and small-scale cycles.
petals:
	@for n in 16 32 64 128 256 512; do \
		python condensation.py -n $$n -d 'petals' -o 'data/petals_n'$$n'.npz'; \
	done 

# Series of annuli of the same size but with different radii. This can
# be used to assess how the condensation works in each case. No. points
# is left at a manageable level.
#
# TODO: make seed configurable?
annulus_series:
	@for n in 128 256; do \
		for r in `seq 0.50 0.05 1.0`; do \
			python condensation.py -r $$r -n $$n -d 'annulus' -o 'data/annulus_n'$$n'_r'$$r'.npz'; \
		done \
	done 

double_annulus:
	@for n in 16 32 64 128 256 512; do \
		python condensation.py -n $$n -d 'double_annulus'; \
	done 

hyperuniform_ellipse:
	@for n in 16 32 64 128 256 512; do \
		python condensation.py -n $$n -d 'hyperuniform_ellipse'; \
	done 

linked_hyperuniform_circles:
	@for n in 16 32 64 128 256 512; do \
		python condensation.py -n $$n -d 'linked_hyperuniform_circles'; \
	done 
