/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package bert

import com.kotlinnlp.utils.DictionarySet
import java.io.File

/**
 * Read the dictionary set for the BERT training from file.
 * Each line of the file must contain a term and its occurrences, separated by a tab char (`\t`).
 *
 * @param filename the filename of the dictionary
 * @param minOccurrences the min number of occurrences to insert a term into the dictionary
 * @param maxTerms the max number of terms to insert into the dictionary or null for no limit
 *
 * @return a dictionary set for the training
 */
internal fun readDictionary(filename: String, minOccurrences: Int, maxTerms: Int? = null): DictionarySet<String> {

  val terms: List<String> = File(filename)
    .readLines()
    .asSequence()
    .map { it.split("\t") }
    .map { it[0] to it[1].toInt() }
    .filter { it.second >= minOccurrences }
    .sortedByDescending { it.second }
    .map { it.first }
    .toList()

  return DictionarySet(terms.take(maxTerms ?: terms.size))
}
