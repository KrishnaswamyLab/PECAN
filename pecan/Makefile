annulus:
	@for n in 16 32 64 128 256 512; do \
		for r in 0.01 0.25 0.50 0.75 0.99; do \
			python condensation.py -r $$r -n $$n -d 'annulus' -o 'annulus_n$$n_r$$r.npz'; \
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
