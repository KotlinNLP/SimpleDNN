/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package deeplearning.multipredictionscorer

import com.kotlinnlp.simplednn.core.layers.types.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.utils.MultiMap
import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionScorer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.context
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 *
 */
class MultiPredictionScorerSpec : Spek({

  describe("a MultiPredictionScorer") {

    context("usage of all networks") {

      on("forward") {

        val scorer = MultiPredictionScorer<DenseNDArray>(model = MultiPredictionScorerUtils.buildModel())
        val outputs = scorer.score(featuresMap = MultiPredictionScorerUtils.buildInputFeaturesMap())

        it("should return the expected output associated to the indices (0, 0)") {
          assertTrue {
            outputs[0, 0]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.309919, 0.510703)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output associated to the indices (0, 1)") {
          assertTrue {
            outputs[0, 1]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.226817, 0.366804)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output associated to the indices (0, 2)") {
          assertTrue {
            outputs[0, 2]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.268098, 1.204324)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output associated to the indices (1, 0)") {
          assertTrue {
            outputs[1, 0]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.031159, -0.360721)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output associated to the indices (1, 1)") {
          assertTrue {
            outputs[1, 1]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.452723, -1.383420)),
              tolerance = 1.0e-06
            )
          }
        }
      }

      on("backward") {

        val scorer = MultiPredictionScorer<DenseNDArray>(model = MultiPredictionScorerUtils.buildModel())
        scorer.score(featuresMap = MultiPredictionScorerUtils.buildInputFeaturesMap())
        scorer.backward(outputsErrors = MultiPredictionScorerUtils.buildOutputErrors(), propagateToInput = true)

        val inputErrors = scorer.getInputErrors(copy = false)
        val paramsErrors = scorer.getParamsErrors(copy = false)

        val errors00In = paramsErrors[0, 0]!!.paramsPerLayer[0] as FeedforwardLayerParameters
        val errors00Out = paramsErrors[0, 0]!!.paramsPerLayer[1] as FeedforwardLayerParameters
        val errors01In = paramsErrors[0, 1]!!.paramsPerLayer[0] as FeedforwardLayerParameters
        val errors01Out = paramsErrors[0, 1]!!.paramsPerLayer[1] as FeedforwardLayerParameters
        val errors02In = paramsErrors[0, 2]!!.paramsPerLayer[0] as FeedforwardLayerParameters
        val errors02Out = paramsErrors[0, 2]!!.paramsPerLayer[1] as FeedforwardLayerParameters
        val errors10In = paramsErrors[1, 0]!!.paramsPerLayer[0] as FeedforwardLayerParameters
        val errors10Out = paramsErrors[1, 0]!!.paramsPerLayer[1] as FeedforwardLayerParameters
        val errors11In = paramsErrors[1, 1]!!.paramsPerLayer[0] as FeedforwardLayerParameters
        val errors11Out = paramsErrors[1, 1]!!.paramsPerLayer[1] as FeedforwardLayerParameters

        it("should return the expected output biases errors associated to the indices (0, 0)") {
          assertTrue {
            errors00Out.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, 0.3)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output weights errors associated to the indices (0, 0)") {
          assertTrue {
            errors00Out.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.446442, -0.552856),
                doubleArrayOf(-0.167416, 0.207321)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input biases errors associated to the indices (0, 0)") {
          assertTrue {
            errors00In.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.399375, 0.120157)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input weights errors associated to the indices (0, 0)") {
          assertTrue {
            errors00In.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.199688, 0.079875),
                doubleArrayOf(-0.060079, 0.024031)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input errors associated to the indices (0, 0)") {
          assertTrue {
            inputErrors[0, 0]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.059734, -0.419516)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output biases errors associated to the indices (0, 1)") {
          assertTrue {
            errors01Out.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.3, -0.9)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output weights errors associated to the indices (0, 1)") {
          assertTrue {
            errors01Out.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.079087, -0.014988),
                doubleArrayOf(0.237262, -0.044963)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input biases errors associated to the indices (0, 1)") {
          assertTrue {
            errors01In.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.362896, -0.418952)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input weights errors associated to the indices (0, 1)") {
          assertTrue {
            errors01In.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.362896, -0.108869),
                doubleArrayOf(-0.418952, -0.125686)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input errors associated to the indices (0, 1)") {
          assertTrue {
            inputErrors[0, 1]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.100607, 0.536082)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output biases errors associated to the indices (0, 2)") {
          assertTrue {
            errors02Out.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.1)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output weights errors associated to the indices (0, 2)") {
          assertTrue {
            errors02Out.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.478386, -0.552856),
                doubleArrayOf(-0.059798, -0.069107)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input biases errors associated to the indices (0, 2)") {
          assertTrue {
            errors02In.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.218422, 0.015673)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input weights errors associated to the indices (0, 2)") {
          assertTrue {
            errors02In.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.131053, -0.196580),
                doubleArrayOf(0.009404, -0.014105)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input errors associated to the indices (0, 2)") {
          assertTrue {
            inputErrors[0, 2]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.05769, -0.204416)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output biases errors associated to the indices (1, 0)") {
          assertTrue {
            errors10Out.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.1)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output weights errors associated to the indices (1, 0)") {
          assertTrue {
            errors10Out.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.343651, -0.123803),
                doubleArrayOf(-0.085913, 0.030951)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input biases errors associated to the indices (1, 0)") {
          assertTrue {
            errors10In.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.005238, 0.217009)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input weights errors associated to the indices (1, 0)") {
          assertTrue {
            errors10In.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.001571, -0.004190),
                doubleArrayOf(0.065103, 0.173607)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input errors associated to the indices (1, 0)") {
          assertTrue {
            inputErrors[1, 0]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.170988, 0.156097)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output biases errors associated to the indices (1, 1)") {
          assertTrue {
            errors11Out.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.2)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output weights errors associated to the indices (1, 1)") {
          assertTrue {
            errors11Out.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.0),
                doubleArrayOf(-0.029777, 0.147044)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input biases errors associated to the indices (1, 1)") {
          assertTrue {
            errors11In.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.117340, -0.036756)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input weights errors associated to the indices (1, 1)") {
          assertTrue {
            errors11In.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.035202, -0.117340),
                doubleArrayOf(-0.011027, 0.036756)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input errors associated to the indices (1, 1)") {
          assertTrue {
            inputErrors[1, 1]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.029265, -0.119601)),
              tolerance = 1.0e-06
            )
          }
        }
      }

      on("backward of (0, 2) and (1, 0) predictions only") {

        val scorer = MultiPredictionScorer<DenseNDArray>(model = MultiPredictionScorerUtils.buildModel())
        scorer.score(featuresMap = MultiPredictionScorerUtils.buildInputFeaturesMap())

        val outputErrorsMap = MultiPredictionScorerUtils.buildOutputErrors()
        scorer.backward(
          outputsErrors = MultiMap(mapOf(
            Pair(0, mapOf(Pair(2, outputErrorsMap[0, 2]!!))),
            Pair(1, mapOf(Pair(0, outputErrorsMap[1, 0]!!)))
          )),
          propagateToInput = true)

        val inputErrors = scorer.getInputErrors(copy = false)
        val paramsErrors = scorer.getParamsErrors(copy = false)

        val errors02In = paramsErrors[0, 2]!!.paramsPerLayer[0] as FeedforwardLayerParameters
        val errors02Out = paramsErrors[0, 2]!!.paramsPerLayer[1] as FeedforwardLayerParameters
        val errors10In = paramsErrors[1, 0]!!.paramsPerLayer[0] as FeedforwardLayerParameters
        val errors10Out = paramsErrors[1, 0]!!.paramsPerLayer[1] as FeedforwardLayerParameters

        it("should return the expected output biases errors associated to the indices (0, 2)") {
          assertTrue {
            errors02Out.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.1)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output weights errors associated to the indices (0, 2)") {
          assertTrue {
            errors02Out.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.478386, -0.552856),
                doubleArrayOf(-0.059798, -0.069107)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input biases errors associated to the indices (0, 2)") {
          assertTrue {
            errors02In.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.218422, 0.015673)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input weights errors associated to the indices (0, 2)") {
          assertTrue {
            errors02In.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.131053, -0.196580),
                doubleArrayOf(0.009404, -0.014105)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input errors associated to the indices (0, 2)") {
          assertTrue {
            inputErrors[0, 2]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.05769, -0.204416)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output biases errors associated to the indices (1, 0)") {
          assertTrue {
            errors10Out.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.1)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output weights errors associated to the indices (1, 0)") {
          assertTrue {
            errors10Out.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.343651, -0.123803),
                doubleArrayOf(-0.085913, 0.030951)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input biases errors associated to the indices (1, 0)") {
          assertTrue {
            errors10In.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.005238, 0.217009)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input weights errors associated to the indices (1, 0)") {
          assertTrue {
            errors10In.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.001571, -0.004190),
                doubleArrayOf(0.065103, 0.173607)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input errors associated to the indices (1, 0)") {
          assertTrue {
            inputErrors[1, 0]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.170988, 0.156097)),
              tolerance = 1.0e-06
            )
          }
        }
      }
    }

    context("usage of the second network only") {

      on("forward") {

        val featuresMap = MultiPredictionScorerUtils.buildInputFeaturesMap()
        val scorer = MultiPredictionScorer<DenseNDArray>(model = MultiPredictionScorerUtils.buildModel())
        val outputs = scorer.score(featuresMap = MultiMap(mapOf(Pair(1, featuresMap[1]!!))))

        it("should return the expected output associated to the indices (1, 0)") {
          assertTrue {
            outputs[1, 0]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.031159, -0.360721)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output associated to the indices (1, 1)") {
          assertTrue {
            outputs[1, 1]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.452723, -1.383420)),
              tolerance = 1.0e-06
            )
          }
        }
      }

      on("backward") {

        val featuresMap = MultiPredictionScorerUtils.buildInputFeaturesMap()
        val scorer = MultiPredictionScorer<DenseNDArray>(model = MultiPredictionScorerUtils.buildModel())
        scorer.score(featuresMap = MultiMap(mapOf(Pair(1, featuresMap[1]!!))))

        val outputErrorsMap = MultiPredictionScorerUtils.buildOutputErrors()
        scorer.backward(outputsErrors = MultiMap(mapOf(Pair(1, outputErrorsMap[1]!!))), propagateToInput = true)

        val inputErrors = scorer.getInputErrors(copy = false)
        val paramsErrors = scorer.getParamsErrors(copy = false)

        val errors10In = paramsErrors[1, 0]!!.paramsPerLayer[0] as FeedforwardLayerParameters
        val errors10Out = paramsErrors[1, 0]!!.paramsPerLayer[1] as FeedforwardLayerParameters
        val errors11In = paramsErrors[1, 1]!!.paramsPerLayer[0] as FeedforwardLayerParameters
        val errors11Out = paramsErrors[1, 1]!!.paramsPerLayer[1] as FeedforwardLayerParameters

        it("should return the expected output biases errors associated to the indices (1, 0)") {
          assertTrue {
            errors10Out.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.1)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output weights errors associated to the indices (1, 0)") {
          assertTrue {
            errors10Out.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.343651, -0.123803),
                doubleArrayOf(-0.085913, 0.030951)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input biases errors associated to the indices (1, 0)") {
          assertTrue {
            errors10In.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.005238, 0.217009)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input weights errors associated to the indices (1, 0)") {
          assertTrue {
            errors10In.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.001571, -0.004190),
                doubleArrayOf(0.065103, 0.173607)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input errors associated to the indices (1, 0)") {
          assertTrue {
            inputErrors[1, 0]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.170988, 0.156097)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output biases errors associated to the indices (1, 1)") {
          assertTrue {
            errors11Out.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.2)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output weights errors associated to the indices (1, 1)") {
          assertTrue {
            errors11Out.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.0),
                doubleArrayOf(-0.029777, 0.147044)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input biases errors associated to the indices (1, 1)") {
          assertTrue {
            errors11In.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.117340, -0.036756)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input weights errors associated to the indices (1, 1)") {
          assertTrue {
            errors11In.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.035202, -0.117340),
                doubleArrayOf(-0.011027, 0.036756)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input errors associated to the indices (1, 1)") {
          assertTrue {
            inputErrors[1, 1]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.029265, -0.119601)),
              tolerance = 1.0e-06
            )
          }
        }
      }

      on("two consecutive backwards") {

        val featuresMap = MultiPredictionScorerUtils.buildInputFeaturesMap()
        val scorer = MultiPredictionScorer<DenseNDArray>(model = MultiPredictionScorerUtils.buildModel())
        val outputErrorsMap = MultiPredictionScorerUtils.buildOutputErrors()

        scorer.score(featuresMap = MultiMap(mapOf(Pair(1, featuresMap[1]!!))))
        scorer.backward(outputsErrors = MultiMap(mapOf(Pair(1, outputErrorsMap[1]!!))), propagateToInput = true)

        scorer.score(featuresMap = MultiMap(mapOf(Pair(1, featuresMap[1]!!))))
        scorer.backward(outputsErrors = MultiMap(mapOf(Pair(1, outputErrorsMap[1]!!))), propagateToInput = true)

        val inputErrors = scorer.getInputErrors(copy = false)
        val paramsErrors = scorer.getParamsErrors(copy = false)

        val errors10In = paramsErrors[1, 0]!!.paramsPerLayer[0] as FeedforwardLayerParameters
        val errors10Out = paramsErrors[1, 0]!!.paramsPerLayer[1] as FeedforwardLayerParameters
        val errors11In = paramsErrors[1, 1]!!.paramsPerLayer[0] as FeedforwardLayerParameters
        val errors11Out = paramsErrors[1, 1]!!.paramsPerLayer[1] as FeedforwardLayerParameters

        it("should return the expected output biases errors associated to the indices (1, 0)") {
          assertTrue {
            errors10Out.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.4, 0.1)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output weights errors associated to the indices (1, 0)") {
          assertTrue {
            errors10Out.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.343651, -0.123803),
                doubleArrayOf(-0.085913, 0.030951)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input biases errors associated to the indices (1, 0)") {
          assertTrue {
            errors10In.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.005238, 0.217009)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input weights errors associated to the indices (1, 0)") {
          assertTrue {
            errors10In.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(-0.001571, -0.004190),
                doubleArrayOf(0.065103, 0.173607)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input errors associated to the indices (1, 0)") {
          assertTrue {
            inputErrors[1, 0]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.170988, 0.156097)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output biases errors associated to the indices (1, 1)") {
          assertTrue {
            errors11Out.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, -0.2)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected output weights errors associated to the indices (1, 1)") {
          assertTrue {
            errors11Out.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.0, 0.0),
                doubleArrayOf(-0.029777, 0.147044)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input biases errors associated to the indices (1, 1)") {
          assertTrue {
            errors11In.unit.biases.values.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(0.117340, -0.036756)),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input weights errors associated to the indices (1, 1)") {
          assertTrue {
            errors11In.unit.weights.values.equals(
              DenseNDArrayFactory.arrayOf(listOf(
                doubleArrayOf(0.035202, -0.117340),
                doubleArrayOf(-0.011027, 0.036756)
              )),
              tolerance = 1.0e-06
            )
          }
        }

        it("should return the expected input errors associated to the indices (1, 1)") {
          assertTrue {
            inputErrors[1, 1]!!.equals(
              DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.029265, -0.119601)),
              tolerance = 1.0e-06
            )
          }
        }
      }

      on("backward with the output errors of all networks") {

        val featuresMap = MultiPredictionScorerUtils.buildInputFeaturesMap()
        val scorer = MultiPredictionScorer<DenseNDArray>(model = MultiPredictionScorerUtils.buildModel())
        scorer.score(featuresMap = MultiMap(mapOf(Pair(1, featuresMap[1]!!))))

        it("should raise an exception") {
          assertFailsWith<IllegalArgumentException> {
            scorer.backward(outputsErrors = MultiPredictionScorerUtils.buildOutputErrors(), propagateToInput = true)
          }
        }
      }
    }
  }
})
