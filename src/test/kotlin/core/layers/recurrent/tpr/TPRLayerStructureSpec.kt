/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.tpr

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

class TPRLayerStructureSpec: Spek({

  describe("a TPRLayer") {

    context("forward") {

      context("without previous state context") {

        val layer = TPRLayerStructureUtils.buildLayer(TPRLayerContextWindow.Empty())
        layer.forward()

        it("should match the expected Symbol Attention Vector") {
          assertTrue {
            layer.aS.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.569546, 0.748381, 0.509998, 0.345246)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected Role Attention Vector") {
          assertTrue {
            layer.aR.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.291109, 0.391740, 0.394126)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected Symbol Vector") {
          assertTrue {
            layer.s.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.142810, 0.913446, 0.425346)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected Role Vector") {
          assertTrue {
            layer.r.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.352204, 0.205093)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected Output vector") {
          assertTrue {
            layer.outputArray.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.050298, 0.029289, 0.321719, 0.187342, 0.149808, 0.087235)),
                tolerance = 0.000001)
          }
        }
      }

      context("with previous state context") {

        val layer = TPRLayerStructureUtils.buildLayer(TPRLayerContextWindow.Back())
        layer.forward()

        it("should match the expected Symbol Attention Vector") {
          assertTrue {
            layer.aS.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.310412, 0.880352, 0.356117, 0.575599)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected Role Attention Vector") {
          assertTrue {
            layer.aR.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.075481, 0.619886, 0.357379)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected Symbol Vector") {
          assertTrue {
            layer.s.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.169241, 0.635193, 0.132934)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected Role Vector") {
          assertTrue {
            layer.r.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.323372, 0.182359)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected Output vector") {
          assertTrue {
            layer.outputArray.values.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.054727, 0.030862, 0.205404, 0.115833, 0.042987, 0.024241)),
                tolerance = 0.000001)
          }
        }
      }

    }

    context("backward") {

      context("without previous and next state") {

        val layer = TPRLayerStructureUtils.buildLayer(TPRLayerContextWindow.Empty())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
            output = layer.outputArray.values,
            outputGold = TPRLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.519701, -0.720710, 0.471719, -1.452657, -0.300191, -0.022764)),
                tolerance = 0.00001)
          }
        }

        it("should match the expected errors of the Binding Matrix") {
          assertTrue {
            layer.bindingMatrix.errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.519701, -0.720710),
                    doubleArrayOf(0.471719, -1.452657),
                    doubleArrayOf(-0.300191, -0.022764)
                )),
                tolerance = 0.00001)
          }
        }

        it("should match the expected errors of the Symbol Vector") {
          assertTrue {
            layer.s.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.330854, -0.131789, -0.110397)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Role Vector") {
          assertTrue {
            layer.r.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.228986, -1.439532)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbol attention Vector") {
          assertTrue {
            layer.aS.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.054416, -0.008955, -0.021861, -0.004433)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Role attention Vector") {
          assertTrue {
            layer.aR.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.070328, -0.052435, -0.018174)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbols embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.s)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.188436752484536, -0.247605225547889, -0.168735193947239, -0.114226262865736),
                    doubleArrayOf(-0.075060214536815, -0.098628856128193, -0.06721247150887, -0.045499870292789),
                    doubleArrayOf(-0.062876419465944, -0.082619392545444, -0.05630252428684, -0.038114318588475)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Roles embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.r)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.06666014408481, 0.089703290625912, 0.090249505767454),
                    doubleArrayOf(-0.419062189806341, -0.563924034648095, -0.567357842307266)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the input -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.043532423752231, 0.04897397672126, -0.04897397672126, -0.005441552969029),
                    doubleArrayOf(0.007164510820115, 0.00806007467263, -0.00806007467263, -0.000895563852514),
                    doubleArrayOf(0.017488998108589, 0.019675122872163, -0.019675122872163, -0.002186124763574),
                    doubleArrayOf(0.003546436137378, 0.00398974065455, -0.00398974065455, -0.000443304517172)
                )),
                tolerance = 0.000001)
          }
        }



        it("should match the expected errors of the input -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.056263184025661, 0.063296082028869, -0.063296082028869, -0.007032898003208),
                    doubleArrayOf(0.041948301479843, 0.047191839164823, -0.047191839164823, -0.00524353768498),
                    doubleArrayOf(0.014539947335829, 0.016357440752808, -0.016357440752808, -0.001817493416979)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the recurrent -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the recurrent -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Roles bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.070328, -0.052435, -0.018174)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbols bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.054416, -0.008955, -0.021861, -0.004433)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.083195466589325, -0.079995855904333, -0.000672136225078, 0.023205789428363)),
                tolerance = 0.000001)
          }
        }

      }

      context("with previous state only") {

        val layer = TPRLayerStructureUtils.buildLayer(TPRLayerContextWindow.Back())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
            output = layer.outputArray.values,
            outputGold = TPRLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.515272047635162, -0.719137223437654, 0.355404003016226, -1.5241663614995, -0.40701259993718, -0.085758082190055)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Binding Matrix") {
          assertTrue {
            layer.bindingMatrix.errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.515272047635162, -0.719137223437654),
                    doubleArrayOf(0.355404003016226, -1.5241663614995),
                    doubleArrayOf(-0.40701259993718, -0.085758082190055)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbol Vector") {
          assertTrue {
            layer.s.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2977662253368, -0.163018516728468, -0.147255383603993)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Role Vector") {
          assertTrue {
            layer.r.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.084438933549135, -1.10124880525326)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbol attention Vector") {
          assertTrue {
            layer.aS.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.052544855142686, -0.008743664227991, -0.028606967248956, 0.009274518622222)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Role attention Vector") {
          assertTrue {
            layer.aR.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.020699695580855, -0.046236444657814, -0.019601819976063)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbols embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.s)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.092430456237147, -0.262139320067201, -0.106039799492167, -0.171394125228089),
                    doubleArrayOf(-0.050603038874782, -0.143513802095011, -0.058053766198113, -0.093833395775674),
                    doubleArrayOf(-0.045709960135667, -0.129636684249611, -0.052440236745614, -0.084760142388163)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Roles embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.r)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.0063735506599, 0.052342525049174, 0.03017676528931),
                    doubleArrayOf(-0.083123563437144, -0.682648877141492, -0.393564026977236)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the input -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf( 0.042035884114149, 0.047290369628418, -0.047290369628418, -0.005254485514269),
                    doubleArrayOf(0.006994931382393, 0.007869297805192, -0.007869297805192, -0.000874366422799),
                    doubleArrayOf(0.022885573799165, 0.02574627052406, -0.02574627052406, -0.002860696724896),
                    doubleArrayOf(-0.007419614897777, -0.00834706676, 0.00834706676, 0.000927451862222)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the input -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.016559756464684, 0.01862972602277, -0.01862972602277, -0.002069969558086),
                    doubleArrayOf(0.036989155726251, 0.041612800192032, -0.041612800192032, -0.004623644465781),
                    doubleArrayOf(0.01568145598085, 0.017641637978456, -0.017641637978456, -0.001960181997606)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the recurrent -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.011086964435107, 0.023697729669352, -0.026219882716201, 0.070042291905201, 0.006118848381366, -0.019231416982223),
                    doubleArrayOf(-0.001844913152106, 0.003943392566824, -0.004363088449767	, 0.011655304415912, 0.00101819969935, -0.003200181107445),
                    doubleArrayOf(-0.00603607008953, 0.012901742229279, -0.014274876657229, 0.038133087342858, 0.003331281336141, -0.010470150013118),
                    doubleArrayOf(0.001956923429289, -0.004182807898622, 0.004627984792489, -0.012362933323422, -0.001080017693558, 0.003394473815733)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the recurrent -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.00436763576756, 0.009335562706966, -0.010329148094847, 0.02759269420928, 0.002410479550391, -0.007576088582593),
                    doubleArrayOf(-0.009755889822799, 0.020852636540674, -0.023071985884249, 0.061633180728866, 0.005384233980402, -0.01692253874476),
                    doubleArrayOf(-0.004135984014949, 0.008840420809204, -0.009781308168055, 0.026129226028091, 0.002282631936212, -0.007174266111239)
                )),
                tolerance = 0.000001)
          }
     }

        it("should match the expected errors of the Roles bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.020699695580855, -0.046236444657814, -0.019601819976063)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbols bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.052544855142686, -0.008743664227991, -0.028606967248956, 0.009274518622222)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.060099369011985, -0.048029952866947, -0.028715724278403, 0.004889227782339)),
                tolerance = 0.000001)
          }
        }

      }

      context("with next state only") {
        val layer = TPRLayerStructureUtils.buildLayer(TPRLayerContextWindow.Front())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
            output = layer.outputArray.values,
            outputGold = TPRLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.249701403353614,  0.279289631931703, 0.98171956873416, -1.74265783974657, 0.129808699976926, 0.747235866903782)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Binding Matrix") {
          assertTrue {
            layer.bindingMatrix.errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.249701403353614, 0.279289631931703),
                    doubleArrayOf(0.98171956873416, -1.74265783974657),
                    doubleArrayOf(0.129808699976926, 0.747235866903782)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbol Vector") {
          assertTrue {
            layer.s.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.030665298337961, -0.011642597310612, 0.198972584038394)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Role Vector") {
          assertTrue {
            layer.r.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.916301710420099, -1.23410486171411)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbol attention Vector") {
          assertTrue {
            layer.aS.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01567489964703, 0.007227245531985, 0.024305171892554, -0.028759718415032)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Role attention Vector") {
          assertTrue {
            layer.aR.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.000875921596484, 0.006486559192793, 0.035967875534)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbols embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.s)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.017465304874356, -0.022949348763758, -0.015639261271837, -0.010587088130656),
                    doubleArrayOf(-0.006630997335104, -0.008713107019294, -0.005937709107433, -0.004019566431043),
                    doubleArrayOf(0.113324083906498, 0.148907444995259, 0.101475752605243, 0.068694596073474)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Roles embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.r)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.266744432797764, 0.358952920168596, 0.361138631737577),
                    doubleArrayOf(-0.359260053328906, -0.483449434688323, -0.48639322191791)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the input -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf( -0.012539919717624, -0.014107409682327, 0.014107409682327, 0.001567489964703),
                    doubleArrayOf(-0.005781796425588, -0.006504520978787, 0.006504520978787, 0.000722724553199),
                    doubleArrayOf(-0.019444137514043,-0.021874654703299, 0.021874654703299, 0.002430517189255),
                    doubleArrayOf(0.023007774732025, 0.025883746573529, -0.025883746573529, -0.002875971841503)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the input -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.000700737277188,  0.000788329436836, -0.000788329436836, -8.75921596484405E-05),
                    doubleArrayOf(-0.005189247354234, -0.005837903273514, 0.005837903273514, 0.000648655919279),
                    doubleArrayOf(-0.0287743004272, -0.0323710879806, 0.0323710879806, 0.0035967875534)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the recurrent -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the recurrent -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Roles bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.000875921596484, 0.006486559192793, 0.035967875534)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbols bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.01567489964703, 0.007227245531985, 0.024305171892554, -0.028759718415032)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.022330619734855, 0.018660116335174, 0.045280243815864, 0.007913525659003)),
                tolerance = 0.000001)
          }
        }
      }

      context("with previous and next state") {
        val layer = TPRLayerStructureUtils.buildLayer(TPRLayerContextWindow.Bilateral())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
            output = layer.outputArray.values,
            outputGold = TPRLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.245272047635161, 0.280862776562346, 0.865404003016226, -1.8141663614995, 0.022987400062821, 0.684241917809945)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Binding Matrix") {
          assertTrue {
            layer.bindingMatrix.errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.245272047635161, 0.280862776562346),
                    doubleArrayOf(0.865404003016226, -1.8141663614995),
                    doubleArrayOf(0.022987400062821, 0.684241917809945)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbol Vector") {
          assertTrue {
            layer.s.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.028096160020637, -0.050982944923792, 0.13221154193128)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Role Vector") {
          assertTrue {
            layer.r.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.511244800638788, -1.01385389080985)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbol attention Vector") {
          assertTrue {
            layer.aS.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.003090437287801, -0.000276646266682, 0.010094898398993, -0.01517018773497)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Role attention Vector") {
          assertTrue {
            layer.aR.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.006956421695052, -0.011947783124287, 0.011811289135905)),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbols embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.s)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.008721408501861, -0.02473453218537, -0.010005537638523, -0.016172138944096),
                    doubleArrayOf(-0.015825760138811, -0.044882976577447, -0.018155925008366, -0.029345763566306),
                    doubleArrayOf(0.041040158690606, 0.11639279662326, 0.047082859653024, 0.076100912884723)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Roles embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.r)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.038589362744448, 0.316913568882586, 0.182708541022483),
                    doubleArrayOf(-0.076526891840165, -0.628473980489548, -0.362330853963468)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the input -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf( -0.00247234983024, -0.002781393559021, 0.002781393559021, 0.00030904372878),
                    doubleArrayOf(0.000221317013345	, 0.000248981640013, -0.000248981640013, -2.7664626668165E-05),
                    doubleArrayOf(-0.008075918719194, -0.009085408559093, 0.009085408559093, 0.001009489839899),
                    doubleArrayOf(0.012136150187976, 0.013653168961473, -0.013653168961473, -0.001517018773497)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the input -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.005565137356041,  0.006260779525546, -0.006260779525546, -0.000695642169505),
                    doubleArrayOf(0.00955822649943, 0.010753004811858, -0.010753004811858, -0.001194778312429),
                    doubleArrayOf(-0.009449031308724, -0.010630160222314, 0.010630160222314, 0.00118112891359)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the recurrent -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(0.000652082267726,-0.001393787216798, 0.001542128206612, -0.004119552904638, -0.000359881422164, 0.001131100047335),
                    doubleArrayOf(-5.83723622698282E-05, 0.000124767466273, -0.000138046487074, 0.000368769473487, 3.22154577550782E-05, -0.000101252533605),
                    doubleArrayOf(0.002130023562187, -0.004552799177946, 0.005037354301097, -0.013456499565857, -0.001175550918563, 0.003694732814031),
                    doubleArrayOf(-0.003200909612079, 0.006841754668471, -0.00756992367975, 0.020221860250715, 0.001766568361737, -0.005552288710999)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the recurrent -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    doubleArrayOf(-0.001467804977656, 0.003137346184468, -0.003471254425831, 0.009272910119504, 0.000810075306389, -0.002546050340389),
                    doubleArrayOf(-0.002520982239225, 0.005388450189053, -0.005961943779019, 0.015926394904675, 0.001391319344823, -0.004372888623489),
                    doubleArrayOf(0.002492182007676, -0.005326891400293, 0.005893833278816, -0.015744448418161, -0.001375424619876, 0.004322931823741)
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Roles bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.006956421695052, -0.011947783124287, 0.011811289135905
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the Symbols bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(0.003090437287801, -0.000276646266682, 0.010094898398993, -0.01517018773497
                )),
                tolerance = 0.000001)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.005503104292945, -0.005218827534345, 0.015450218964438, -0.000914123430928)),
                tolerance = 0.000001)
          }
        }
      }
    }
  }
})
