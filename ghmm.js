class GHMM {
	constructor(n_hidden, out_dim){
		this.n_hidden = n_hidden;
		this.out_dim = out_dim;

		this.cov = [];

		var i;
		for (i = 0; i < n_hidden; i++) { 
    		this.cov[i] = math.identity(out_dim, out_dim);
		}

		this.mean = [math.matrix([0, 0]), math.matrix([5, 0]), math.matrix([10, 0])];
		this.A = math.add(math.zeros(n_hidden, n_hidden), 1/n_hidden);
		this.pi = math.add(math.zeros(1, n_hidden), 1/n_hidden);
		this.old_alpha = math.add(math.zeros(1, n_hidden), 1/n_hidden);
	}

	p(y){
		var toReturn = math.ones(1, this.n_hidden);

		var i;
		for(i = 0; i < this.n_hidden; i++){
			var cov = this.cov[i];
			var adjusted_y = math.subtract(y, this.mean[i]);
			var exponent = math.chain(-.5)
							.multiply(math.multiply(
													adjusted_y, 
													math.multiply(cov,
																  adjusted_y)))
							.done();
			var divisor = math.sqrt(math.pow(math.pi * 2, this.out_dim) * math.det(cov));

			toReturn = math.subset(toReturn, math.index(0, i),  math.exp(exponent) / divisor);
		}

		return toReturn;
	}

	propagate(Y, n){
		this.alpha = math.zeros(n , this.n_hidden);
		this.gamma = math.zeros(n, this.n_hidden);
		this.zai = math.zeros(n, this.n_hidden, this.n_hidden);
		this.Y = Y;

		
		window.alert(this.alpha);
	}
}