/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package utils

import com.kotlinnlp.simplednn.dataset.Corpus
import com.kotlinnlp.simplednn.dataset.Example
import utils.exampleextractor.ExampleExtractor
import java.io.FileInputStream
import CorpusPaths
import com.beust.klaxon.JsonArray
import com.beust.klaxon.Parser

/**
 * A helper to read corpora from file (containing training, validation and test sets).
 */
class CorpusReader<ExampleType : Example> {

  /**
   * Read the corpus from the given [corpusPath], extracting examples with the an [exampleExtractor] from a pure JSON
   * file if [perLine] is false, otherwise from a JSON-line file.
   *
   * @param corpusPath the [CorpusPaths] from which to read the datasets
   * @param exampleExtractor an [ExampleExtractor]
   * @param perLine a Boolean indicating if the file contains a JSON object per line, or a unique pure JSON
   *
   * @return the read [Corpus]
   */
  fun read(corpusPath: CorpusPaths,
           exampleExtractor: ExampleExtractor<ExampleType>,
           perLine: Boolean): Corpus<ExampleType> {

    println("\n-- CORPUS READING")

    val startTime = System.currentTimeMillis()

    val dataset = Corpus(
      training = this.readDataset(corpusPath.training, exampleExtractor, perLine = perLine),
      validation = this.readDataset(corpusPath.validation, exampleExtractor, perLine = perLine),
      test = this.readDataset(corpusPath.test, exampleExtractor, perLine = perLine))

    println("Elapsed time: %s s".format(System.currentTimeMillis() - startTime))
    println("Train: %d examples".format(dataset.training.size))
    println("Validation: %d examples".format(dataset.validation.size))
    println("Test: %d examples".format(dataset.test.size))

    return dataset
  }

  /**
   * Read a dataset from the given file extracting examples with the given [exampleExtractor].
   *
   * @param filename the name of the dataset file
   * @param exampleExtractor an [ExampleExtractor]
   * @param perLine a Boolean indicating if the file contains a JSON object per line, or a unique pure JSON
   *
   * @return the read dataset
   */
  private fun readDataset(filename: String,
                          exampleExtractor: ExampleExtractor<ExampleType>,
                          perLine: Boolean): ArrayList<ExampleType> {
    return if (perLine)
      this.readDatasetPerLine(filename = filename, exampleExtractor = exampleExtractor)
    else
      this.readDatasetFromWholeFile(filename = filename, exampleExtractor = exampleExtractor)
  }

  /**
   * Read a dataset per line from the given file extracting examples with the given [exampleExtractor].
   *
   * @param filename the name of the dataset file
   * @param exampleExtractor an [ExampleExtractor]
   *
   * @return the read dataset
   */
  private fun readDatasetPerLine(filename: String,
                                 exampleExtractor: ExampleExtractor<ExampleType>): ArrayList<ExampleType> {

    val examples = ArrayList<ExampleType>()
    val file = FileInputStream(filename)
    val jsonParser = Parser()

    file.reader().forEachLine {
      examples.add(exampleExtractor.extract(jsonParser.parse(StringBuilder(it)) as JsonArray<*>))
    }

    return examples
  }

  /**
   * Read a dataset from the given JSON file extracting examples with the given [exampleExtractor].
   *
   * @param filename the name of the dataset file
   * @param exampleExtractor an [ExampleExtractor]
   *
   * @return the read dataset
   */
  private fun readDatasetFromWholeFile(filename: String,
                                       exampleExtractor: ExampleExtractor<ExampleType>): ArrayList<ExampleType> {

    val examples = ArrayList<ExampleType>()

    (Parser().parse(filename) as JsonArray<*>).forEach {
      examples.add(exampleExtractor.extract(it as JsonArray<*>))
    }

    return examples
  }
}
