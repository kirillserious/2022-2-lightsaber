package main

import (
	"fmt"

	progress "github.com/cheggaaa/pb/v3"
)

const T_START = 0.0
const T_FINAL = 2.0

const SEQUENCE_N = 10 // разбиение времени
const SEQUENCE_M = 3  // размерность вектора последовательности

// precalculated: 2**3
const SEQUENCE_DIM = 8

// precalculated: (2**3)**10
const SEQUENCE_ITERATIONS = 1073741824

func SliceCopy[T any](in []T) (out []T) {
	out = make([]T, len(in))
	copy(out, in)
	return
}

func IntegerPow(x, y int) (res int) {
	res = x
	for i := 1; i < y; i++ {
		res *= x
	}
	return res
}

func Sequence(length, max int) <-chan []int {
	ch := make(chan []int)

	go func() {
		defer close(ch)
		sequence := make([]int, length)

		iterations := IntegerPow(max, length)
		bar := progress.StartNew(iterations)
		for i := 0; i < iterations; i++ {
			ch <- sequence
			sequence = SliceCopy(sequence)
			for j := 0; j < length; j++ {
				if sequence[j] == max-1 {
					sequence[j] = 0
				} else {
					sequence[j]++
					break
				}
			}
			bar.Increment()
		}
	}()
	return ch
}

func DecodeControl(control int) []float64 {
	u := make([]float64, 3)
	if control&1 == 0 {
		u[0] = -1
	} else {
		u[0] = 1
	}
	if control&2 == 0 {
		u[1] = -1
	} else {
		u[1] = 1
	}
	if control&4 == 0 {
		u[2] = -1
	} else {
		u[2] = 1
	}
	return u
}

var (
	g       float64 = 9.8
	m               = []float64{0.1, 0.1, 0.1}
	l               = []float64{1, 1, 1}
	I               = []float64{0, 0, 0}
	z_start         = []float64{0, 0, 0}
	z_final         = []float64{1, 1, 1}
)

func init() {
	for i := 0; i < 3; i++ {
		I[i] = m[i] * l[i] * l[i] / 3
	}
}

func IterativeCost(u []float64) float64 {
	return 0.01 * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2])
}

func FinalConst(z []float64) float64 {
	e1 := z[0] - z_final[0]
	e2 := z[1] - z_final[1]
	e3 := z[2] - z_final[2]
	return e1*e1 + e2*e2 + e3*e3
}

func main() {
	cost_delta_t := float64(0.001)
	cost_iterations := int((T_FINAL - T_START) / cost_delta_t)
	min_cost := float64(1000000000)
	min_sequence := make([]int, 6)

	for sequence := range Sequence(6, IntegerPow(2, 3)) {
		z := make([]float64, 6)
		copy(z, z_start)
		cost := float64(0)
		for i := 0; i < cost_iterations; i++ {
			control_index := i * len(sequence) / cost_iterations
			control := DecodeControl(sequence[control_index])
			//control[0] = z[0] * control[0]
			//control[1] = z[1] * control[1]
			//control[2] = z[2] * control[2]
			z = fk(z, control, cost_delta_t)
			cost += cost_delta_t * IterativeCost(control)
		}
		cost += 100 * FinalConst(z)
		//fmt.Printf("%v -> %v\n", sequence, z)
		//fmt.Println(cost)

		if cost < min_cost {
			min_cost = cost
			copy(min_sequence, sequence)
		}
	}
	fmt.Println(min_cost)
	fmt.Println(min_sequence)
}
