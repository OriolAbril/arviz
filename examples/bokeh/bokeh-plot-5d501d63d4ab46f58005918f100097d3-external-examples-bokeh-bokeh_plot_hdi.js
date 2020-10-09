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
    
      
      
    
      var element = document.getElementById("eb1cfe89-dc45-442e-851a-185aa31db52a");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'eb1cfe89-dc45-442e-851a-185aa31db52a' but no matching script tag was found.")
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
                    
                  var docs_json = '{"1acbbab4-8969-44d6-af97-81698efd2fa8":{"roots":{"references":[{"attributes":{},"id":"5243","type":"LinearScale"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"5261","type":"BoxAnnotation"},{"attributes":{"formatter":{"id":"5285"},"ticker":{"id":"5250"}},"id":"5249","type":"LinearAxis"},{"attributes":{},"id":"5253","type":"ResetTool"},{"attributes":{},"id":"5290","type":"UnionRenderers"},{"attributes":{},"id":"5285","type":"BasicTickFormatter"},{"attributes":{},"id":"5246","type":"BasicTicker"},{"attributes":{},"id":"5287","type":"BasicTickFormatter"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[400]}},"selected":{"id":"5291"},"selection_policy":{"id":"5290"}},"id":"5272","type":"ColumnDataSource"},{"attributes":{"data":{"x":{"__ndarray__":"RVpTMTneAsAzteF8Vf/+vwpM+o/m3/6/RpKOT1QF/r8WMJ5gA3f9v+5gT5I3LP2/qbBIA/cC/b+Z3X1bVLb7v0XhJZ/Kn/q/t4P+xVBy97+JNqkBGXL2vyDjU6gkBfa/RcbILMan9L8PY7FISYTzv++Smz9eZvO/h/3VsV9c87/LVV0oil7xvwGUjBNnx/C/F7e+jUx/8L+/I6hCKfnuv1aNWTtn2u6/tlyMjvwF7r83joS7WATtv6cwPr3VgOy/zwmilsWi679gsHP695Xrv5LnMF7RNeu/UHgSLotT6L/ldm9+sN3lvzwKgRWWRuW/A0Ac22285L9Yv0Yg5jzkv93bu1Li/eO/841C0nea478U3gUdYKHhv/BB8tMwRd+/d3e7dMH53r+Ee9gMpTvev978RPdRY92/f5SpH7Un3b/1en9C6MLbv+xHMo3yD9C/5mVaocf4x79Sxsv+ohrEv0PQfedXnMC/FmZhF3quvL+RiG+Hlam3v/s9ydcG9LO/NNFc7Sxdlr/hw/XmYh6Wv++CmSToWpW/g0hscid9ib8OG/nvP16Fv6/jlijxwLM//p/05ftavD8TBoEsvf/APx+gW/4tOsE/p94jG50/wT8fTJOtaAXDP4tlSZPnNMM/RzsimFKzxD+2zdFFCFrFP+o0HkSHkss/sW8LB6oZ0z/Sa6DlAsvTPwk3rLJTbNU/mbc/z9VM2D+leUMaLUXZP5t0SUWR6t0/cgCiTFvC3j+A10+39abgP4mhdw16wOA/gCiD/AjW4D8LQa7+rJzkP5P7plzxjuU/7NETjgn/5T8RDpK+AWPmPwJ1mIVsZeg/NiL5YfQY6T+s/5+5Yb3rP7v6fgikEuw/nZOp03D97T9Bwifl0jHvP1OntOwaGPE/T78WSp628T+4XE9n3NzxPw+0dcD2ffI/EgyJ67NV9D/Fp3aWO9v0P8AGB6XfCvc/ieM4KqTh9z9ggr+rxD75P5l8JBS4f/k/B0e/vggY/T+/fRHozUX/P/FYwotAvP8/FIwwCnEvAEAzgA+ThJsAQNQevHJG6QFAKnVtcxvOAkA=","dtype":"float64","order":"little","shape":[100]},"y":{"__ndarray__":"dktZnY1D6j9mJY9BVYDwP/vZArgMkPA/3bY42FX98D/157BPfkTxP4lP2DbkafE/rKdbfoR+8T80EUHS1STyP14PbbAasPI/JL4AnddG9D+8ZCt/88b0P3AO1qtt/fQ/3pyb6Rys9T94Tqdb2z32P4g2MuDQTPY/PAEVJ9BR9j8aVdHrulD3PwC2OXZMnPc/dKQguVnA9z8Q91WvtUH4P6qcKTFmSfg/0uhc3IB++D9y3B7R6b74P9ZzsJDK3/g/jH1Xmk4X+T/oE2MBghr5PxzGc6iLMvk/7GF7NB3r+T9HImTgk4j6P3G9n3parvo//+84ieTQ+j8qUO53xvD6PwkJUWuHAPs/g1xvC2IZ+z97iL74p5f7P8K3geVZF/w/EZFo0ccg/D+Q8GReizj8P2RgF8GVU/w/cM0KXAlb/D+hELD3oof8PwK3Wa4B/v0/olnqhXOA/j+bQxPQVb7+P/wiiIE69v4/z/REL4wa/z+7g8RTs0L/PxC2QclfYP8/XkYlpkXT/z94FDI6w9P/P/rMti9K1f8/t5ON2ILm/z/lBhDAoer/P49bosQDTwBAgNKX72txAEAxCGTp/YcAQAHd8m/RiQBA9R7Z6PyJAEBhmmxFK5gAQCxLmjynmQBA2hHBlJqlAEBuji5C0KoAQKfxIDqU3ABA+7ZwoJoxAUC9BlousDwBQHHDKjvFVgFAevvzXM2EAUCaN6TRUpQBQEqXVBSp3gFAByDKtCXsAUDw+um23hQCQDH0rkEPGAJAEGWQH8EaAkAhyNWflZMCQHLflCvesQJAPnrCMeG/AkDCQdI3YMwCQKAOs5CtDANARyQ/jB4jA0D2/zM3rHcDQFffD4FUggNAdDJ1Gq6/A0BI+KRcOuYDQNUpLbsGRgRA1K+FkqdtBEAu19MZN3cEQARtHbB9nwRABEPi+mwVBUDxqZ3lzjYFQLDBQem3wgVA4jiOCmn4BUCY4O8qsU8GQCYfCQXuXwZAwtGvLwJGB0BwXwR6c9EHQDyW8CIQ7wdACkYYhbgXCEAawIdJwk0IQGoPXjmj9AhAlbq2uQ1nCUA=","dtype":"float64","order":"little","shape":[100]}},"selected":{"id":"5293"},"selection_policy":{"id":"5292"}},"id":"5277","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"5277"},"glyph":{"id":"5278"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5279"},"selection_glyph":null,"view":{"id":"5281"}},"id":"5280","type":"GlyphRenderer"},{"attributes":{"formatter":{"id":"5287"},"ticker":{"id":"5246"}},"id":"5245","type":"LinearAxis"},{"attributes":{},"id":"5254","type":"PanTool"},{"attributes":{"below":[{"id":"5245"}],"center":[{"id":"5248"},{"id":"5252"}],"left":[{"id":"5249"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"5275"},{"id":"5280"}],"title":{"id":"5283"},"toolbar":{"id":"5263"},"toolbar_location":"above","x_range":{"id":"5237"},"x_scale":{"id":"5241"},"y_range":{"id":"5239"},"y_scale":{"id":"5243"}},"id":"5236","subtype":"Figure","type":"Plot"},{"attributes":{"overlay":{"id":"5261"}},"id":"5255","type":"BoxZoomTool"},{"attributes":{"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5278","type":"Line"},{"attributes":{},"id":"5239","type":"DataRange1d"},{"attributes":{"text":""},"id":"5283","type":"Title"},{"attributes":{},"id":"5258","type":"UndoTool"},{"attributes":{},"id":"5250","type":"BasicTicker"},{"attributes":{"callback":null},"id":"5260","type":"HoverTool"},{"attributes":{"axis":{"id":"5245"},"ticker":null},"id":"5248","type":"Grid"},{"attributes":{},"id":"5292","type":"UnionRenderers"},{"attributes":{},"id":"5259","type":"SaveTool"},{"attributes":{"axis":{"id":"5249"},"dimension":1,"ticker":null},"id":"5252","type":"Grid"},{"attributes":{"source":{"id":"5277"}},"id":"5281","type":"CDSView"},{"attributes":{},"id":"5256","type":"WheelZoomTool"},{"attributes":{"line_alpha":0.1,"line_width":3,"x":{"field":"x"},"y":{"field":"y"}},"id":"5279","type":"Line"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"5262","type":"PolyAnnotation"},{"attributes":{},"id":"5241","type":"LinearScale"},{"attributes":{},"id":"5293","type":"Selection"},{"attributes":{"fill_alpha":0.1,"fill_color":"#ff0000","line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5274","type":"Patch"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"5253"},{"id":"5254"},{"id":"5255"},{"id":"5256"},{"id":"5257"},{"id":"5258"},{"id":"5259"},{"id":"5260"}]},"id":"5263","type":"Toolbar"},{"attributes":{"source":{"id":"5272"}},"id":"5276","type":"CDSView"},{"attributes":{},"id":"5291","type":"Selection"},{"attributes":{"fill_alpha":0.5,"fill_color":"#ff0000","line_alpha":0,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"5273","type":"Patch"},{"attributes":{"data_source":{"id":"5272"},"glyph":{"id":"5273"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"5274"},"selection_glyph":null,"view":{"id":"5276"}},"id":"5275","type":"GlyphRenderer"},{"attributes":{},"id":"5237","type":"DataRange1d"},{"attributes":{"overlay":{"id":"5262"}},"id":"5257","type":"LassoSelectTool"}],"root_ids":["5236"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"1acbbab4-8969-44d6-af97-81698efd2fa8","root_ids":["5236"],"roots":{"5236":"eb1cfe89-dc45-442e-851a-185aa31db52a"}}];
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