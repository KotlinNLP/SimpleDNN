/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.kotlin.KotlinModule
import java.io.File
import java.io.FileNotFoundException
import java.io.InputStream

/**
 *
 * @param training training dataset file path
 * @param validation validation dataset file path
 * @param test test dataset file path
 */
data class CorpusPaths(val training: String, val validation: String, val test: String)

/**
 *
 */
data class ExampleConfiguration(val datasets_paths: CorpusPaths)

/**
 *
 */
data class Configuration(val mnist: ExampleConfiguration,
                         val mnist_sequence: ExampleConfiguration,
                         val sparse_input: ExampleConfiguration,
                         val progressive_sum: ExampleConfiguration,
                         val han_classifier: ExampleConfiguration,
                         val vectors_average: ExampleConfiguration){

  /**
   *
   */
  companion object{

    /**
     *
     */
    private val defaultConfigurationStream: InputStream =
      Charsets.javaClass.getResourceAsStream(File.separator + "configuration.yaml")
        ?: throw FileNotFoundException(File.separator + "configuration.yaml")

    /**
     *
     */
    fun loadFromFile(inputStream: InputStream = defaultConfigurationStream): Configuration {

      val mapper = ObjectMapper(YAMLFactory()) // Enable YAML parsing

      mapper.registerModule(KotlinModule()) // Enable Kotlin support

      return mapper.readValue(inputStream, Configuration::class.java)
    }
  }
}
