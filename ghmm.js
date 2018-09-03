class GHMM {
	constructor(n_hidden, out_dim, cov_guess, mean_guess){
		this.n_hidden = n_hidden;
		this.out_dim = out_dim;

		this.cov = [];
		this.mean = [];

		var i;
		for (i = 0; i < n_hidden; i++) { 
    		this.cov[i] = math.multiply(math.identity(out_dim, out_dim), cov_guess[i]);
    		this.mean[i] = math.matrix(mean_guess[i]);
		}

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
		window.alert(this.alpha);

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

	sum_columns(matrix){
		var m = matrix._size[0];
		var n = matrix._size[1];
		var output = math.ones(1, n);

		var i;
		var curr_sum;
		for(i = 0; i < n; i++){
			curr_sum = math.sum(math.subset(matrix, math.index([0, m - 1], i)));
			output = math.subset(output, math.index(0,i), curr_sum);
		}

		return math.squeeze(output);
	}

	update(n){
		this.pi = math.subset(this.gamma, math.index(0, math.range(0, this.n_hidden)));

		var i;
		for(i = 0; i < this.n_hidden; i++){
			var curr_gamma = math.squeeze(math.subset(this.gamma, math.index(math.range(0, n), i)));
			var temp_numerator = math.multiply(math.transpose(curr_gamma), this.Y);
			var temp_denominator = math.sum(curr_gamma);

			this.mean[i] = math.divide(temp_numerator, temp_denominator);
		}

		for(i = 0; i < this.n_hidden; i++){
			//this will help to subtract the mean from each element in each row
			var temp_mean_matrix = math.multiply(math.ones(n, this.out_dim), math.diag(this.mean[i]));

			var Y_shifted = math.subtract(this.Y, temp_mean_matrix);

			var curr_gamma = math.squeeze(math.subset(this.gamma, math.index(math.range(0, n), i)));

			var temp_numerator = math.multiply(math.transpose(Y_shifted), math.multiply(math.diag(curr_gamma), Y_shifted));
			var temp_denominator = math.sum(curr_gamma);

			this.cov[i] = math.divide(temp_numerator, temp_denominator);

		}

		var temp_sum_zai = this.zai;
		var temp_sum_gamma = this.sum_columns(this.gamma);
		this.A = math.dotDivide(temp_sum_zai, math.multiply(math.ones(this.n_hidden, this.n_hidden), math.diag(temp_sum_gamma)));

		for(i = 0; i < this.A._size[1]; i++){
			var temp_sum = math.sum(math.subset(this.A, math.index(math.range(0, this.A._size[0]), i)));
			var temp_normalized_row = math.divide(math.subset(this.A, math.index(math.range(0, this.A._size[0]), i)), temp_sum);
			this.A = math.subset(this.A, math.index(math.range(0, this.A._size[0]), i), temp_normalized_row);
		}
	}

	train(Y){
		window.alert("training");
		var i;
		for(i = 0; i < 100; i++){
			this.propagate(Y, Y._size[0]);
			this.update(Y._size[0]);
		}
	}

	evaluate(y){
		this.old_alpha = math.dotMultiply(this.p(y), math.multiply(this.old_alpha, this.A));
		this.old_alpha = math.divide(this.old_alpha, math.sum(this.old_alpha));
		return this.old_alpha; 
	}
}























