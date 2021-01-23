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
    
      
      
    
      var element = document.getElementById("a08be54d-8f09-41e8-8cfa-bbebb83de89b");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'a08be54d-8f09-41e8-8cfa-bbebb83de89b' but no matching script tag was found.")
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
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
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
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
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
                    
                  var docs_json = '{"4a18ef22-d21c-4eb9-b544-e4ec621be033":{"roots":{"references":[{"attributes":{},"id":"16859","type":"DataRange1d"},{"attributes":{},"id":"16863","type":"LinearScale"},{"attributes":{"formatter":{"id":"16928"},"ticker":{"id":"16870"}},"id":"16869","type":"LinearAxis"},{"attributes":{"formatter":{"id":"16926"},"ticker":{"id":"16866"}},"id":"16865","type":"LinearAxis"},{"attributes":{},"id":"16878","type":"HelpTool"},{"attributes":{},"id":"16866","type":"BasicTicker"},{"attributes":{"axis":{"id":"16865"},"ticker":null},"id":"16868","type":"Grid"},{"attributes":{"axis":{"id":"16869"},"dimension":1,"ticker":null},"id":"16872","type":"Grid"},{"attributes":{"text":""},"id":"16943","type":"Title"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"16910","type":"BoxAnnotation"},{"attributes":{},"id":"16870","type":"BasicTicker"},{"attributes":{},"id":"16888","type":"DataRange1d"},{"attributes":{},"id":"16861","type":"LinearScale"},{"attributes":{},"id":"16956","type":"UnionRenderers"},{"attributes":{},"id":"16874","type":"WheelZoomTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"16873"},{"id":"16874"},{"id":"16875"},{"id":"16876"},{"id":"16877"},{"id":"16878"}]},"id":"16880","type":"Toolbar"},{"attributes":{},"id":"16957","type":"Selection"},{"attributes":{},"id":"16873","type":"PanTool"},{"attributes":{"overlay":{"id":"16879"}},"id":"16875","type":"BoxZoomTool"},{"attributes":{},"id":"16876","type":"SaveTool"},{"attributes":{},"id":"16877","type":"ResetTool"},{"attributes":{},"id":"16892","type":"LinearScale"},{"attributes":{},"id":"16930","type":"UnionRenderers"},{"attributes":{"text":""},"id":"16924","type":"Title"},{"attributes":{"formatter":{"id":"16949"},"ticker":{"id":"16897"}},"id":"16896","type":"LinearAxis"},{"attributes":{},"id":"16949","type":"BasicTickFormatter"},{"attributes":{},"id":"16890","type":"DataRange1d"},{"attributes":{},"id":"16931","type":"Selection"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"16919","type":"Quad"},{"attributes":{"formatter":{"id":"16951"},"ticker":{"id":"16901"}},"id":"16900","type":"LinearAxis"},{"attributes":{},"id":"16857","type":"DataRange1d"},{"attributes":{},"id":"16894","type":"LinearScale"},{"attributes":{},"id":"16909","type":"HelpTool"},{"attributes":{},"id":"16897","type":"BasicTicker"},{"attributes":{},"id":"16926","type":"BasicTickFormatter"},{"attributes":{"axis":{"id":"16896"},"ticker":null},"id":"16899","type":"Grid"},{"attributes":{"axis":{"id":"16900"},"dimension":1,"ticker":null},"id":"16903","type":"Grid"},{"attributes":{},"id":"16901","type":"BasicTicker"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11],"right":[1,2,3,4,5,6,7,8,9,10,11,12],"top":{"__ndarray__":"nMQgsHJokT8bL90kBoG1PzMzMzMzM8M/F9nO91PjxT/ufD81XrrJPzvfT42XbsI/AAAAAAAAwD+kcD0K16OwP3npJjEIrJw/eekmMQisfD/6fmq8dJN4P/yp8dJNYmA/","dtype":"float64","order":"little","shape":[12]}},"selected":{"id":"16931"},"selection_policy":{"id":"16930"}},"id":"16918","type":"ColumnDataSource"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"16935","type":"Line"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"16904"},{"id":"16905"},{"id":"16906"},{"id":"16907"},{"id":"16908"},{"id":"16909"}]},"id":"16911","type":"Toolbar"},{"attributes":{"source":{"id":"16934"}},"id":"16938","type":"CDSView"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"16921"}]},"id":"16933","type":"LegendItem"},{"attributes":{"below":[{"id":"16896"}],"center":[{"id":"16899"},{"id":"16903"}],"left":[{"id":"16900"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"16937"}],"title":{"id":"16943"},"toolbar":{"id":"16911"},"x_range":{"id":"16888"},"x_scale":{"id":"16892"},"y_range":{"id":"16890"},"y_scale":{"id":"16894"}},"id":"16887","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"16905","type":"WheelZoomTool"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"16957"},"selection_policy":{"id":"16956"}},"id":"16934","type":"ColumnDataSource"},{"attributes":{},"id":"16904","type":"PanTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"16879","type":"BoxAnnotation"},{"attributes":{"data_source":{"id":"16918"},"glyph":{"id":"16919"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"16920"},"selection_glyph":null,"view":{"id":"16922"}},"id":"16921","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"16910"}},"id":"16906","type":"BoxZoomTool"},{"attributes":{},"id":"16951","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"16918"}},"id":"16922","type":"CDSView"},{"attributes":{"data_source":{"id":"16934"},"glyph":{"id":"16935"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"16936"},"selection_glyph":null,"view":{"id":"16938"}},"id":"16937","type":"GlyphRenderer"},{"attributes":{},"id":"16907","type":"SaveTool"},{"attributes":{"children":[{"id":"16856"},{"id":"16887"}]},"id":"16939","type":"Row"},{"attributes":{"below":[{"id":"16865"}],"center":[{"id":"16868"},{"id":"16872"},{"id":"16932"}],"left":[{"id":"16869"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"16921"}],"title":{"id":"16924"},"toolbar":{"id":"16880"},"x_range":{"id":"16857"},"x_scale":{"id":"16861"},"y_range":{"id":"16859"},"y_scale":{"id":"16863"}},"id":"16856","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"16908","type":"ResetTool"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"16920","type":"Quad"},{"attributes":{"items":[{"id":"16933"}]},"id":"16932","type":"Legend"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"16936","type":"Line"},{"attributes":{},"id":"16928","type":"BasicTickFormatter"}],"root_ids":["16939"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"4a18ef22-d21c-4eb9-b544-e4ec621be033","root_ids":["16939"],"roots":{"16939":"a08be54d-8f09-41e8-8cfa-bbebb83de89b"}}];
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