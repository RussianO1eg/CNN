<?
/*
 * В этом примере рассказывается об использовании многослойного персептрона Theano
 *
 * Многослойный перцептрон это логистической регрессор, где вместо подачи на
 * вход логистической регрессии добавляется промежуточный слой, называемый
 * скрытым слоем, который имеет нелинейную функцию активации (обычно тангенсальную или
 * сигмовидной). Можно использовать много скрытых слоев.
 * В руководстве также решается проблема MNIST классификации цифр.
 *
 * Матемическа формула: f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x)))
 *
 * Цитата из учебника "Распознавание образов и машинное обучение" Christopher M. Bishop, section 5
 */

class HiddenLayer {
	var $rng		= Array();		// Случайное число, используемое для инициализации весов
	var $input;						// A symbolic tensor of shape (n_examples, n_in)
	var $n_in		= 1;			// Рамзерность входа
	var $n_out		= 1;			// Количество скрытых выходов
	var $W			= false;
	var $b			= false;
	var $activation	= "sigmoid";	// Нелинейность используемая в скрытом слое
	var $output		= false;

	function Init($input, $n_in, $n_out, $W=false, $b=false, $activation="sigmoid") {
		/*
		 * Обычно скрытые слои мнослойного персептрона - это блоки с сигмоидальной
		 * функцией активации. Матрица весов W составляет форму ($n_in, $n_out)
		 * и смещение вектора B имеет форму ($n_out).
		 *
		 * ПРИМЕЧАНИЕ: Исполдьзуемая здесь нелинейность достигается использованием
		 * функции tanh (гиперболический тангенс.
		 *
		 * Скрытые блоки активации определяются по формуле: tanh(dot($input,$W) + $b)
		 */
		$this->input		= $input;
		$this->n_in			= $n_in;
		$this->n_out		= $n_out;
		$this->W			= $W;
		$this->b			= $b;
		$this->activation	= $activation;

		if(!$this->W) {
			$low	= -sqrt(6./($this->n_in + $this->n_out));	// Минимальный вес
			$hight	= sqrt(6./($this->n_in + $this->n_out));	// Максимальный вес

			for($i=0; $i<$this->n_in; $i++) {
				for ($j=0; $j<$this->n_out; $j++) {
					$this->W[$i][$j] = rand($low*100, $hight*100)/100;

					if($this->activation === "sigmoid") {
						$this->W[$i][$j] *= 4;
					}
				}
			}
		}

		if(!$this->b) {
			$this->b = Array();
			for($i=0; $i<$this->n_out; $i++) {
				$this->b[$i] = 0;
			}
		}

		
	}
}

/* Test code */
$obj = new HiddenLayer;
$obj->Init(false, 16, 500);
echo "<pre>";
print_r($obj->W);
echo "</pre>";
?>
