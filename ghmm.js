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
		this.zai = math.zeros(this.n_hidden, this.n_hidden);
		this.Y = Y;		

		this.alpha = math.subset(this.alpha, math.index(0, [0,1,2]), math.dotMultiply(this.pi, this.p(math.squeeze(math.subset(Y, math.index(0, [0, this.out_dim - 1]))))));
		this.alpha = math.subset(this.alpha, math.index(0, [0,1,2]), math.divide(math.subset(this.alpha, math.index(0, [0,1,2])), math.sum(this.alpha)));
		var i;
		for(i = 1; i < n; i++){
			this.alpha = math.subset(this.alpha, math.index(i, math.range(0,this.n_hidden)), math.dotMultiply(this.p(math.squeeze(math.subset(Y, math.index(i, math.range(0,this.out_dim))))), math.multiply(math.subset(this.alpha, math.index(0, math.range(0,this.n_hidden))), this.A)));
			this.alpha = math.subset(this.alpha, math.index(i, math.range(0, this.n_hidden)), math.divide(math.subset(this.alpha, math.index(i, math.range(0, this.n_hidden))), math.sum(math.subset(this.alpha, math.index(i, math.range(0,this.n_hidden))))));
		}
		this.gamma = math.subset(this.gamma, math.index(n - 1, math.range(0, this.n_hidden)), math.subset(this.alpha, math.index(n - 1, math.range(0, this.n_hidden))));
		var j;
		for(j = n - 2; j > -1; j--){
			this.gamma = math.subset(this.gamma, math.index(j, math.range(0, this.n_hidden)), math.dotDivide(math.dotMultiply(math.subset(this.alpha, math.index(j, math.range(0, this.n_hidden))),  math.multiply(math.subset(this.gamma, math.index(j+1, math.range(0, this.n_hidden))), math.transpose(this.A))), math.multiply(math.subset(this.alpha, math.index(j, math.range(0, this.n_hidden))), this.A)));	
		}

		var k;
		for (k = 0; k < n - 1; k++){
			this.zai = math.add(
						 math.multiply(
				         math.multiply(
									   math.transpose(math.subset(this.alpha, math.index(k, math.range(0, this.n_hidden)))),
									   math.dotDivide(
									 			      math.dotMultiply(math.subset(this.gamma, math.index(k + 1, math.range(0, this.n_hidden))), this.p(math.squeeze(math.subset(this.Y, math.index(k + 1, math.range(0, this.out_dim)))))),
									 				  math.add(math.subset(this.alpha, math.index(k + 1, math.range(0, this.n_hidden))), .0000000000000001))),
				         			   this.A),
						 this.zai);
		}
	}
	evaluate(y){
		this.old_alpha = math.dotMultiply(this.p(y), math.multiply(this.old_alpha, this.A));
		this.old_alpha = math.divide(this.old_alpha, math.sum(this.old_alpha));
		return this.old_alpha; 
	}
}























