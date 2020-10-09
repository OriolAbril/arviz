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
    
      
      
    
      var element = document.getElementById("cfa9b081-9be7-4586-9dc1-d644924ef22d");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'cfa9b081-9be7-4586-9dc1-d644924ef22d' but no matching script tag was found.")
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
                    
                  var docs_json = '{"4adc66e6-1ead-441f-a571-220d20ec6677":{"roots":{"references":[{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"67624","type":"BoxAnnotation"},{"attributes":{"formatter":{"id":"67641"},"ticker":{"id":"67584"}},"id":"67583","type":"LinearAxis"},{"attributes":{"axis":{"id":"67583"},"dimension":1,"ticker":null},"id":"67586","type":"Grid"},{"attributes":{},"id":"67670","type":"UnionRenderers"},{"attributes":{},"id":"67621","type":"SaveTool"},{"attributes":{"below":[{"id":"67579"}],"center":[{"id":"67582"},{"id":"67586"},{"id":"67646"}],"left":[{"id":"67583"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"67635"}],"title":{"id":"67637"},"toolbar":{"id":"67594"},"x_range":{"id":"67571"},"x_scale":{"id":"67575"},"y_range":{"id":"67573"},"y_scale":{"id":"67577"}},"id":"67570","subtype":"Figure","type":"Plot"},{"attributes":{"source":{"id":"67632"}},"id":"67636","type":"CDSView"},{"attributes":{},"id":"67619","type":"WheelZoomTool"},{"attributes":{},"id":"67591","type":"ResetTool"},{"attributes":{"overlay":{"id":"67624"}},"id":"67620","type":"BoxZoomTool"},{"attributes":{},"id":"67571","type":"DataRange1d"},{"attributes":{},"id":"67606","type":"LinearScale"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"67635"}]},"id":"67647","type":"LegendItem"},{"attributes":{"axis":{"id":"67579"},"ticker":null},"id":"67582","type":"Grid"},{"attributes":{},"id":"67641","type":"BasicTickFormatter"},{"attributes":{},"id":"67608","type":"LinearScale"},{"attributes":{"children":[{"id":"67570"},{"id":"67601"}]},"id":"67653","type":"Row"},{"attributes":{},"id":"67604","type":"DataRange1d"},{"attributes":{},"id":"67618","type":"PanTool"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11],"right":[1,2,3,4,5,6,7,8,9,10,11,12],"top":{"__ndarray__":"exSuR+F6lD/D9Shcj8K1P5MYBFYOLcI/lkOLbOf7yT9iEFg5tMjGPyPb+X5qvMQ/QmDl0CLbuT9U46WbxCCwP5qZmZmZmZk/ObTIdr6fij956SYxCKx8P/yp8dJNYlA/","dtype":"float64","order":"little","shape":[12]}},"selected":{"id":"67643"},"selection_policy":{"id":"67644"}},"id":"67632","type":"ColumnDataSource"},{"attributes":{},"id":"67615","type":"BasicTicker"},{"attributes":{"items":[{"id":"67647"}]},"id":"67646","type":"Legend"},{"attributes":{"formatter":{"id":"67664"},"ticker":{"id":"67615"}},"id":"67614","type":"LinearAxis"},{"attributes":{},"id":"67662","type":"BasicTickFormatter"},{"attributes":{},"id":"67587","type":"PanTool"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"67669"},"selection_policy":{"id":"67670"}},"id":"67648","type":"ColumnDataSource"},{"attributes":{},"id":"67590","type":"SaveTool"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"67650","type":"Line"},{"attributes":{},"id":"67643","type":"Selection"},{"attributes":{},"id":"67602","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"67618"},{"id":"67619"},{"id":"67620"},{"id":"67621"},{"id":"67622"},{"id":"67623"}]},"id":"67625","type":"Toolbar"},{"attributes":{},"id":"67669","type":"Selection"},{"attributes":{},"id":"67639","type":"BasicTickFormatter"},{"attributes":{"formatter":{"id":"67662"},"ticker":{"id":"67611"}},"id":"67610","type":"LinearAxis"},{"attributes":{"below":[{"id":"67610"}],"center":[{"id":"67613"},{"id":"67617"}],"left":[{"id":"67614"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"67651"}],"title":{"id":"67656"},"toolbar":{"id":"67625"},"x_range":{"id":"67602"},"x_scale":{"id":"67606"},"y_range":{"id":"67604"},"y_scale":{"id":"67608"}},"id":"67601","subtype":"Figure","type":"Plot"},{"attributes":{"axis":{"id":"67614"},"dimension":1,"ticker":null},"id":"67617","type":"Grid"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"67649","type":"Line"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"67633","type":"Quad"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"67634","type":"Quad"},{"attributes":{},"id":"67644","type":"UnionRenderers"},{"attributes":{},"id":"67664","type":"BasicTickFormatter"},{"attributes":{"text":""},"id":"67637","type":"Title"},{"attributes":{},"id":"67584","type":"BasicTicker"},{"attributes":{},"id":"67577","type":"LinearScale"},{"attributes":{},"id":"67611","type":"BasicTicker"},{"attributes":{"source":{"id":"67648"}},"id":"67652","type":"CDSView"},{"attributes":{"data_source":{"id":"67632"},"glyph":{"id":"67633"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"67634"},"selection_glyph":null,"view":{"id":"67636"}},"id":"67635","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"67648"},"glyph":{"id":"67649"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"67650"},"selection_glyph":null,"view":{"id":"67652"}},"id":"67651","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"67610"},"ticker":null},"id":"67613","type":"Grid"},{"attributes":{"overlay":{"id":"67593"}},"id":"67589","type":"BoxZoomTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"67587"},{"id":"67588"},{"id":"67589"},{"id":"67590"},{"id":"67591"},{"id":"67592"}]},"id":"67594","type":"Toolbar"},{"attributes":{"text":""},"id":"67656","type":"Title"},{"attributes":{},"id":"67580","type":"BasicTicker"},{"attributes":{},"id":"67573","type":"DataRange1d"},{"attributes":{"formatter":{"id":"67639"},"ticker":{"id":"67580"}},"id":"67579","type":"LinearAxis"},{"attributes":{},"id":"67592","type":"HelpTool"},{"attributes":{},"id":"67622","type":"ResetTool"},{"attributes":{},"id":"67588","type":"WheelZoomTool"},{"attributes":{},"id":"67623","type":"HelpTool"},{"attributes":{},"id":"67575","type":"LinearScale"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"67593","type":"BoxAnnotation"}],"root_ids":["67653"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"4adc66e6-1ead-441f-a571-220d20ec6677","root_ids":["67653"],"roots":{"67653":"cfa9b081-9be7-4586-9dc1-d644924ef22d"}}];
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