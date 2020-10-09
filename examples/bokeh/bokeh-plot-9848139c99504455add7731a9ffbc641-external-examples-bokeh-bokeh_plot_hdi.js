(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("715b09d7-0d77-4281-ad2d-ec9e30477c96");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '715b09d7-0d77-4281-ad2d-ec9e30477c96' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"23b35e84-53e1-4cf7-ba00-70c3c2bd80af":{"roots":{"references":[{"attributes":{},"id":"5378","type":"BasicTickFormatter"},{"attributes":{},"id":"5384","type":"UnionRenderers"},{"attributes":{"axis":{"id":"5341"},"dimension":1,"ticker":null},"id":"5344","type":"Grid"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5370","type":"Line"},{"attributes":{"callback":null},"id":"5352","type":"HoverTool"},{"attributes":{},"id":"5351","type":"SaveTool"},{"attributes":{"below":[{"id":"5337"}],"center":[{"id":"5340"},{"id":"5344"}],"left":[{"id":"5341"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5367"},{"id":"5372"}],"title":{"id":"5375"},"toolbar":{"id":"5355"},"toolbar_location":"above","x_range":{"id":"5329"},"x_scale":{"id":"5333"},"y_range":{"id":"5331"},"y_scale":{"id":"5335"}},"id":"5328","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"5376","type":"BasicTickFormatter"},{"attributes":{},"id":"5350","type":"UndoTool"},{"attributes":{"source":{"id":"5364"}},"id":"5368","type":"CDSView"},{"attributes":{},"id":"5385","type":"Selection"},{"attributes":{"data_source":{"id":"5369"},"glyph":{"id":"5370"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5371"},"selection_glyph":null,"view":{"id":"5373"}},"id":"5372","type":"GlyphRenderer"},{"attributes":{},"id":"5329","type":"DataRange1d"},{"attributes":{"formatter":{"id":"5378"},"ticker":{"id":"5338"}},"id":"5337","type":"LinearAxis"},{"attributes":{},"id":"5331","type":"DataRange1d"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5365","type":"Patch"},{"attributes":{"axis":{"id":"5337"},"ticker":null},"id":"5340","type":"Grid"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5371","type":"Line"},{"attributes":{"data_source":{"id":"5364"},"glyph":{"id":"5365"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5366"},"selection_glyph":null,"view":{"id":"5368"}},"id":"5367","type":"GlyphRenderer"},{"attributes":{},"id":"5342","type":"BasicTicker"},{"attributes":{},"id":"5335","type":"LinearScale"},{"attributes":{"text":""},"id":"5375","type":"Title"},{"attributes":{"formatter":{"id":"5376"},"ticker":{"id":"5342"}},"id":"5341","type":"LinearAxis"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5383"},"selection_policy":{"id":"5382"}},"id":"5364","type":"ColumnDataSource"},{"attributes":{},"id":"5338","type":"BasicTicker"},{"attributes":{},"id":"5383","type":"Selection"},{"attributes":{"source":{"id":"5369"}},"id":"5373","type":"CDSView"},{"attributes":{},"id":"5348","type":"WheelZoomTool"},{"attributes":{},"id":"5346","type":"PanTool"},{"attributes":{},"id":"5345","type":"ResetTool"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5366","type":"Patch"},{"attributes":{},"id":"5333","type":"LinearScale"},{"attributes":{"overlay":{"id":"5353"}},"id":"5347","type":"BoxZoomTool"},{"attributes":{"data":{"x":{"__ndarray__":"eq3yZkdsA8BTQwjtOusCwEApKGG2kgLAw0WloIGPAMB8iyooQGgAwNTZ5lFcVfq/N1w6fTpS+r9TVW7BN0P5v/FOp640qPa/oA5wOJjY8r8WhWKuThnyv/182bIfyvG/vCNDDTyC77/4VR7HZe3uv77EksRL4e2/GOE0Lof96b8fwE9pygzov+AJkzIDuea//+902lkO5r+5cjDDtnrkvyFrMvSkCeS/bdCIsD5Q47/tJTFHf1nhv5ClkLPykd6/y/CB0ozQ3L+BnQO9isbbv3P37cDC9tq/fMl7+8/T079L/QFH7MvTv04eR0ElM9O/Iwq4l3980b+yH6hl8tPQvygiE9bIdM+/vDlHNhH5yL/Q0XnRSszIv83WoEoUTsi/gXT8utUByL+f7Qt6N6DHv5WYwHRWUsS/T2/Icnjewb9NnNi+GtDBv+bq46xj47y/JatDPrVavL+xOz+Jr3+7vz7vZGoUzLm/FcLPCqL2uL/eShKlWr2mv1xl/Vcp1Ju/dOKyKD1GeL+HzHKpPudwv9khikyqRkO/9w+a43u3hz+qoxn2ZieQP08XRFCkiZ8/ydNiXohKoT8/26sDu9amP/B2HMs0oqg/Kj4LQhGksj9s8qxSxwuzP0HjwZcp5bQ/Sl1KUtCWtj9K8gvyEzLBP8lWBdGBkck/3+8pRkYHyz9TE+XRQkzTP/0O880Al9Q/ki8K6jld1T8rsXwl63DVP/QEJ1+PRNc/+QwZfVt61z9dRMHZjvzXP5Sb/ZknmNo/XrOG5X/l2j+MreBZRtHbP7xiZWMY1N8/2wFY5Oo+4D9rAo3T6kDgP8/kI/qYsOE/i+v0DKC54T/Bt/IvFpbiP8qW01+MMOU/O/By7dyT5T8+LM9c1T7rPxQF0NnAVes/lt6BfVZs7j99aDFPbZ7uP1knzWc8Cu8/fUE7U2Tn8D+hHfK7MZ/yP9op+kDZp/I/NMGqIYRS8z+2CIXB1qHzP2cZtH3FA/U/W0kotkhh+D8tFhlNOVD6P3kcD2l/Nvw/0UwbCW/Q/D9aG8fG17f9P/5SsN5vqwNAoBJPYP5YBUA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"DKUaMnEn6T9aee8liinqP4Ctrz2T2uo/enS1vvzg7j8I6aqvfy/vPxaTDNdR1fI/5NFiweLW8j9W1UgfZF7zP4hYrKjlq/Q/sPjH47OT9j91vc6oWPP2P4JBkybwGvc/ETev/HAf+D+CajiOpkT4P9BO2w6th/g/usdyNJ6A+T/4D6xlzfz5P4g9WzO/Ufo/AMRiiWl8+j9S4zNPUuH6Pzhl88KW/fo/5cvdU/Ar+z+FtjMuoKn7P07rjanBLfw/58GvZe5l/D9QjF+oLof8PxJB4qcnofw/0IaQAIaF/T9XwB93gob9PzYc11ebmf0/vP4IDXDQ/T8K/EqzgeX9P97NnnKzCP4/ZIyb7G5w/j/jYuhSO3P+P5PyVbsee/4/uDhQpOJ//j8mQV+I/IX+P3f2s5jauv4/C3nTeBji/j87dhJU/uL+P6ngmOLkGP8/p+INViod/z8iBraDAiT/P4bYrFyfMf8/74Gp70o4/z/VtmuVCqX/PzUFUK1XyP8/j6Zr4dzz/z+aRqtgjPf/P143W5XL/v8/CM3xvdsLAECkGfZmJxAAQBdEUKSJHwBAqMW8EJUiAEC2Vwd2rS0AQO44lmlEMQBA+SwIRZBKAEDKs0odL0wAQI0HX6aUUwBAdSlJQVtaAECSX5CfkIkAQLYqiA6MzABAf08xMjrYAEA1UR4txDQBQPAw3wxwSQFA+aKgntNVAUATy1eyDlcBQE9w8vVIdAFA0JDRt6V3AUBGFJztyH8BQLnZn3mCqQFANmtY/leuAUDZCp5lFL0BQCxWNoZB/QFAOwCLXN0HAkBNoHFaHQgCQJp8RB8TNgJAcZ2eATQ3AkD4Vv7FwlICQNly+osRpgJAB16unXuyAkCI5Zmr2mcDQKIAOhu4agNA0zuwz4rNA0AQLeapzdMDQOuk+YxH4QNAX9DOFNk5BEBoh/xuzKcEQHaKPlD2qQRATbBqCKHUBEAuQmGwdegEQFoGbV/xQAVAVxKKLVIYBkCLRUZTDpQGQB7HQ9qfDQdANNNGwhs0B0DWxrHx9W0HQH8pWO+31QlAUIknMH+sCkA=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5385"},"selection_policy":{"id":"5384"}},"id":"5369","type":"ColumnDataSource"},{"attributes":{},"id":"5382","type":"UnionRenderers"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5353","type":"BoxAnnotation"},{"attributes":{"overlay":{"id":"5354"}},"id":"5349","type":"LassoSelectTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5345"},{"id":"5346"},{"id":"5347"},{"id":"5348"},{"id":"5349"},{"id":"5350"},{"id":"5351"},{"id":"5352"}]},"id":"5355","type":"Toolbar"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5354","type":"PolyAnnotation"}],"root_ids":["5328"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"23b35e84-53e1-4cf7-ba00-70c3c2bd80af","root_ids":["5328"],"roots":{"5328":"715b09d7-0d77-4281-ad2d-ec9e30477c96"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();