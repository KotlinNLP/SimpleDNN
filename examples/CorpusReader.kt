/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.jsoniter.JsonIterator
import com.kotlinnlp.simplednn.dataset.Corpus
import com.kotlinnlp.simplednn.dataset.Example
import java.io.BufferedInputStream
import java.io.FileInputStream
import java.util.concurrent.TimeUnit

/**
 *
 */
class CorpusReader<ExampleType : Example> {

  /**
   *
   */
  fun read(corpusPath: CorpusPaths,
           examplesExtractor: ExampleExtractor<ExampleType>,
           perLine: Boolean): Corpus<ExampleType> {

    println("\n-- CORPUS READING")

    val startTime = System.nanoTime()

    val dataset = Corpus(
      training = this.readDataset(corpusPath.training, examplesExtractor, perLine = perLine),
      validation = this.readDataset(corpusPath.validation, examplesExtractor, perLine = perLine),
      test = this.readDataset(corpusPath.test, examplesExtractor, perLine = perLine))

    val elapsedTime = System.nanoTime() - startTime

    println("Elapsed time: ${TimeUnit.MILLISECONDS.convert(elapsedTime, TimeUnit.NANOSECONDS) / 1000.0}s")
    println("Train: %d examples".format(dataset.training.size))
    println("Validation: %d examples".format(dataset.validation.size))
    println("Test: %d examples".format(dataset.test.size))

    return dataset
  }

  /**
   *
   */
  private fun readDataset(filename: String,
                          extractExample: ExampleExtractor<ExampleType>,
                          perLine: Boolean): ArrayList<ExampleType> {
    return if (perLine)
      this.readDatasetPerLine(filename = filename, examplesExtractor = extractExample)
    else
      this.readDatasetFromWholeFile(filename = filename, examplesExtractor = extractExample)
  }

  /**
   *
   */
  private fun readDatasetPerLine(filename: String,
                                 examplesExtractor: ExampleExtractor<ExampleType>): ArrayList<ExampleType> {

    val examples = ArrayList<ExampleType>()
    val file = FileInputStream(filename)

    file.reader().forEachLine {
      examples.add(examplesExtractor.extract(JsonIterator.parse(it)))
    }

    return examples
  }

  /**
   *
   */
  private fun readDatasetFromWholeFile(filename: String,
                                       examplesExtractor: ExampleExtractor<ExampleType>): ArrayList<ExampleType> {

    val examples = ArrayList<ExampleType>()
    val iterator = JsonIterator.parse(BufferedInputStream(FileInputStream(filename)), 2048)

    while(iterator.readArray()) {
      examples.add(examplesExtractor.extract(iterator))
    }

    return examples
  }
}
