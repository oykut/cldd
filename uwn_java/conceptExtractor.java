// Extract concepts of terms using UWN Java API


import org.lexvo.uwn.Entity;
import org.lexvo.uwn.Statement;
import org.lexvo.uwn.UWN;

import java.io.*;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

class conceptExtractor {


    public static void main(String[] args) throws Exception {


        String language = "fr";
        String topic = "title_en";
        String lang2 = "eng";
//        String lang1 = "fra";
//        String lang1 = "spa";
        String lang1 = "fra";

        String path = "/Users/oyku/Desktop/blocked/vocabulary/" + lang1 + "/";

        String filepath = "/Users/oyku/Desktop/blocked/lemmatized/" + language + "/" + topic + "_lemmatized.txt";
        String outpath = "/Users/oyku/Desktop/blocked/wordnet/" + language + "/" + topic + "_wordnet.txt";

        HashMap<String, HashMap<String, Float>> map = new HashMap<String, HashMap<String, Float>>();

        String[] plgns = {"uwn", "wordnet"};

        // Instantiate UWN, providing a pointer to the plugins directory.
        UWN uwn = new UWN(new File("/Users/oyku/Desktop/uwnapi/plugins/"), plgns);


        BufferedReader br = new BufferedReader(new FileReader(filepath));

        String st;
        int countx = 0;
        while ((st = br.readLine()) != null) {
            HashMap<String, Float> entity = new HashMap<String, Float>();
            countx += 1;
            String[] data = st.split(",");

            String key = data[0];
            String lemmatized = data[1];

            //Checking for German
            int count = 0;
            Iterator<Statement> it = uwn.getMeanings(Entity.createTerm(lemmatized, lang1));
            while (it.hasNext()) {
                count += 1;
                Statement meaningStatement = it.next();
                Entity meaning = meaningStatement.getObject();
                entity.put(meaning.toString(), meaningStatement.getWeight());
            }

            //Checking for English
            int count1 = 0;
            Iterator<Statement> it1 = uwn.getMeanings(Entity.createTerm(lemmatized, lang2));
            while (it1.hasNext()) {
                count1 += 1;
                Statement meaningStatement = it1.next();
                Entity meaning = meaningStatement.getObject();
                entity.put(meaning.toString(), meaningStatement.getWeight());
            }

            if ((count != 0) || (count1 != 0)) {
                map.put(key, entity);
            }
        }

        System.out.println(countx);
        System.out.println(map.size());


        ///////////////////////
        BufferedWriter writer = new BufferedWriter(new FileWriter(outpath));

        for (Map.Entry<String, HashMap<String, Float>> pair : map.entrySet()) {
            String key = pair.getKey();
            String newstr = key;
            HashMap<String, Float> entity = pair.getValue();
            for (Map.Entry<String, Float> pair1 : entity.entrySet()) {
                newstr = newstr + "," + pair1.getKey() + "," + pair1.getValue();
            }

            writer.write(newstr + "\n");
        }
        writer.close();


    }

}
