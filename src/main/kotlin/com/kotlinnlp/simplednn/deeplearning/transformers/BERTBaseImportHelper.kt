/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.transformers

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.layers.models.feedforward.norm.NormLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.DictionarySet
import com.kotlinnlp.utils.getLinesCount
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.File

/**
 * Import the parameters of a [BERTModel] from a file of named matrices.
 * Build a new [BERTModel] with the parameters read from file.
 */
object BERTBaseImportHelper {

  /**
   * The regex that matches the index of a BERT layer in its param name.
   */
  private val LAYER_INDEX_REGEX = Regex("^bert\\.encoder\\.layer\\.(\\d+)")

  /**
   * Build a [BERTModel] with the parameters read from file.
   *
   * @param params the parameters of the model, associated by name
   * @param vocab the vocabulary used to train the model
   * @param numOfHeads the number of attention heads of the model
   *
   * @return a new BERT model with the given parameters
   */
  fun buildModel(params: Map<String, DenseNDArray>, vocab: DictionarySet<String>, numOfHeads: Int): BERTModel {

    val embMatrix: DenseNDArray = params.getValue("bert.embeddings.word_embeddings.weight")
    val vocabSize: Int = embMatrix.shape.dim1
    val inputSize: Int = embMatrix.shape.dim2

    require(inputSize % numOfHeads == 0) {
      "The number of attention heads ($numOfHeads) must be an exact divider of the input size ($inputSize)"
    }

    val wordEmb = EmbeddingsMap<String>(size = inputSize).apply {
      val embeddings: List<DenseNDArray> = embMatrix.getRows()
      vocab.getElements().forEach { form ->
        set(key = form, embedding = ParamsArray(embeddings[vocab.getId(form)!!]))
      }
    }

    return BERTModel(
      inputSize = inputSize,
      attentionSize = inputSize / numOfHeads,
      attentionOutputSize = inputSize / numOfHeads,
      outputHiddenSize = params.getValue("bert.encoder.layer.0.intermediate.dense.bias").shape.dim1,
      numOfHeads = numOfHeads,
      numOfLayers = countLayers(params),
      vocabulary = vocab,
      wordEmbeddings = wordEmb,
      weightsInitializer = null,
      biasesInitializer = null
    ).apply {

      params.getValue("bert.embeddings.position_embeddings.weight").getRows().forEachIndexed { i, values ->
        positionalEmb.set(key = i, embedding = ParamsArray(values))
      }

      params.getValue("bert.embeddings.token_type_embeddings.weight").getRows().forEachIndexed { i, values ->
        tokenTypeEmb.getOrSet(i).values.assignValues(values)
      }

      classifierModel = buildClassifierModel(hiddenSize = inputSize, outputSize = vocabSize)

      getAssignMap(this).forEach { (paramName, array) ->
        array.values.assignValues(params.getValue(paramName))
      }
    }
  }

  /**
   * Read the parameters of a BERT model from a file of named matrices.
   *
   * @param filename the input filename
   * @param numOfHeads the number of heads in which to split the multi-head attention parameters
   *
   * @return the parameters associated by name
   */
  fun readParams(filename: String, numOfHeads: Int): Map<String, DenseNDArray> {

    val progress = ProgressIndicatorBar(total = getLinesCount(filename))
    val paramsMap: MutableMap<String, DenseNDArray> = mutableMapOf()

    var firstLine = true
    var paramName = ""
    val values: MutableList<DoubleArray> = mutableListOf()

    File(filename).forEachLine { line ->

      progress.tick()

      when {

        firstLine -> {
          firstLine = false
          paramName = line.trim()
          values.clear()
        }

        line.isBlank() -> {
          firstLine = true
          paramsMap[paramName] = DenseNDArrayFactory.arrayOf(values)
        }

        else -> values.add(line.split("\t").map { it.toDouble() }.toTypedArray().toDoubleArray())
      }
    }

    return expandAttentionParams(params = paramsMap, numOfHeads = numOfHeads)
  }

  /**
   * Expand the parameters map read from file splitting horizontally the multi-head attention parameters by a given
   * number of heads.
   *
   * @param params the parameters read from file
   * @param numOfHeads the number of heads in which to split the multi-head attention parameters
   *
   * @return a copy of the parameters map with the split parameters added
   */
  private fun expandAttentionParams(params: Map<String, DenseNDArray>, numOfHeads: Int): Map<String, DenseNDArray> {

    val paramsCopy: MutableMap<String, DenseNDArray> = params.toMutableMap()

    (0 until countLayers(params)).forEach { i ->

      val oldPrefix = "bert.encoder.layer.$i.attention.self"
      val dim: Int = params.getValue("$oldPrefix.query.bias").shape.dim1 / numOfHeads

      val qwRows: List<DenseNDArray> = params.getValue("$oldPrefix.query.weight").getRows()
      val kwRows: List<DenseNDArray> = params.getValue("$oldPrefix.key.weight").getRows()
      val vwRows: List<DenseNDArray> = params.getValue("$oldPrefix.value.weight").getRows()

      (0 until numOfHeads).forEach { j ->

        val newPrefix = "bert.encoder.layer.$i.$j.attention.self"

        val from = j * dim
        val to = (j + 1) * dim

        paramsCopy["$newPrefix.query.weight"] = DenseNDArrayFactory.fromRows(qwRows.subList(from, to))
        paramsCopy["$newPrefix.query.bias"] = params.getValue("$oldPrefix.query.bias").getRange(from, to)
        paramsCopy["$newPrefix.key.weight"] = DenseNDArrayFactory.fromRows(kwRows.subList(from, to))
        paramsCopy["$newPrefix.key.bias"] = params.getValue("$oldPrefix.key.bias").getRange(from, to)
        paramsCopy["$newPrefix.value.weight"] = DenseNDArrayFactory.fromRows(vwRows.subList(from, to))
        paramsCopy["$newPrefix.value.bias"] = params.getValue("$oldPrefix.value.bias").getRange(from, to)
      }
    }

    return paramsCopy
  }

  /**
   * Get the map of parameters names to the related arrays of a [BERTModel].
   *
   * @param model a BERT model
   *
   * @return the assignment map of the given BERT model
   */
  private fun getAssignMap(model: BERTModel): Map<String, ParamsArray> {

    val assignMap: MutableMap<String, ParamsArray> = mutableMapOf(
      "bert.embeddings.LayerNorm.weight" to (model.embNorm.paramsPerLayer.single() as NormLayerParameters).g,
      "bert.embeddings.LayerNorm.bias" to (model.embNorm.paramsPerLayer.single() as NormLayerParameters).b,
      "cls.predictions.transform.dense.weight" to
        (model.classifierModel!!.paramsPerLayer[0] as FeedforwardLayerParameters).unit.weights,
      "cls.predictions.transform.dense.bias" to
        (model.classifierModel!!.paramsPerLayer[0] as FeedforwardLayerParameters).unit.biases,
      "cls.predictions.transform.LayerNorm.weight" to
        (model.classifierModel!!.paramsPerLayer[1] as NormLayerParameters).g,
      "cls.predictions.transform.LayerNorm.bias" to
        (model.classifierModel!!.paramsPerLayer[1] as NormLayerParameters).b,
      "cls.predictions.decoder.weight" to
        (model.classifierModel!!.paramsPerLayer[2] as FeedforwardLayerParameters).unit.weights,
      "cls.predictions.decoder.bias" to
        (model.classifierModel!!.paramsPerLayer[2] as FeedforwardLayerParameters).unit.biases
    )

    model.layers.forEachIndexed { i, layerParams ->
      assignMap += this.getLayerAssignMap(params = layerParams, i = i, numOfHeads = model.numOfHeads)
    }

    return assignMap
  }

  /**
   * @param params the BERT layer parameters
   * @param i the layer index
   * @param numOfHeads the number of self-attention heads
   *
   * @return the assignment map of the given BERT layer parameters
   */
  private fun getLayerAssignMap(params: BERTParameters, i: Int, numOfHeads: Int): Map<String, ParamsArray> {

    val assignMap: MutableMap<String, ParamsArray> = mutableMapOf()

    (0 until numOfHeads).forEach { j ->
      assignMap += mapOf(
        "bert.encoder.layer.$i.$j.attention.self.query.weight" to params.attention.attention[j].queries.weights,
        "bert.encoder.layer.$i.$j.attention.self.query.bias" to params.attention.attention[j].queries.biases,
        "bert.encoder.layer.$i.$j.attention.self.key.weight" to params.attention.attention[j].keys.weights,
        "bert.encoder.layer.$i.$j.attention.self.key.bias" to params.attention.attention[j].keys.biases,
        "bert.encoder.layer.$i.$j.attention.self.value.weight" to params.attention.attention[j].values.weights,
        "bert.encoder.layer.$i.$j.attention.self.value.bias" to params.attention.attention[j].values.biases
      )
    }

    assignMap += mapOf(
      "bert.encoder.layer.$i.attention.output.dense.weight" to params.attention.merge.output.unit.weights,
      "bert.encoder.layer.$i.attention.output.dense.bias" to params.attention.merge.output.unit.biases,
      "bert.encoder.layer.$i.attention.output.LayerNorm.weight" to
        (params.multiHeadNorm.paramsPerLayer.single() as NormLayerParameters).g,
      "bert.encoder.layer.$i.attention.output.LayerNorm.bias" to
        (params.multiHeadNorm.paramsPerLayer.single() as NormLayerParameters).b,
      "bert.encoder.layer.$i.intermediate.dense.weight" to
        (params.outputFF.paramsPerLayer[0] as FeedforwardLayerParameters).unit.weights,
      "bert.encoder.layer.$i.intermediate.dense.bias" to
        (params.outputFF.paramsPerLayer[0] as FeedforwardLayerParameters).unit.biases,
      "bert.encoder.layer.$i.output.dense.weight" to
        (params.outputFF.paramsPerLayer[1] as FeedforwardLayerParameters).unit.weights,
      "bert.encoder.layer.$i.output.dense.bias" to
        (params.outputFF.paramsPerLayer[1] as FeedforwardLayerParameters).unit.biases,
      "bert.encoder.layer.$i.output.LayerNorm.weight" to
        (params.outputNorm.paramsPerLayer.single() as NormLayerParameters).g,
      "bert.encoder.layer.$i.output.LayerNorm.bias" to
        (params.outputNorm.paramsPerLayer.single() as NormLayerParameters).b
    )

    return assignMap
  }

  /**
   * @param params the parameters read from file, associated by name
   *
   * @return the number of BERT layers defined in given parameters
   */
  private fun countLayers(params: Map<String, DenseNDArray>): Int =
    params.keys
      .asSequence()
      .mapNotNull { LAYER_INDEX_REGEX.find(it) }
      .map { it.groupValues[1].toInt() }
      .toSet()
      .count()
}
