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
    
      
      
    
      var element = document.getElementById("05280ebc-0f00-448b-8659-14c35b70e79d");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '05280ebc-0f00-448b-8659-14c35b70e79d' but no matching script tag was found.")
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
                    
                  var docs_json = '{"ed527f2a-c25d-4d8d-bf65-4f65897e9b3a":{"roots":{"references":[{"attributes":{},"id":"17401","type":"LinearScale"},{"attributes":{"text":"mu"},"id":"17441","type":"Title"},{"attributes":{"axis":{"id":"17403"},"ticker":null},"id":"17406","type":"Grid"},{"attributes":{"toolbar":{"id":"17456"},"toolbar_location":"above"},"id":"17457","type":"ToolbarBox"},{"attributes":{},"id":"17397","type":"DataRange1d"},{"attributes":{"axis":{"id":"17407"},"dimension":1,"ticker":null},"id":"17410","type":"Grid"},{"attributes":{},"id":"17408","type":"BasicTicker"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"17432","type":"Circle"},{"attributes":{"children":[[{"id":"17394"},0,0]]},"id":"17455","type":"GridBox"},{"attributes":{"overlay":{"id":"17419"}},"id":"17413","type":"BoxZoomTool"},{"attributes":{},"id":"17450","type":"UnionRenderers"},{"attributes":{},"id":"17412","type":"PanTool"},{"attributes":{},"id":"17411","type":"ResetTool"},{"attributes":{},"id":"17451","type":"Selection"},{"attributes":{},"id":"17417","type":"SaveTool"},{"attributes":{},"id":"17445","type":"BasicTickFormatter"},{"attributes":{},"id":"17414","type":"WheelZoomTool"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"17435","type":"Dash"},{"attributes":{"overlay":{"id":"17420"}},"id":"17415","type":"LassoSelectTool"},{"attributes":{},"id":"17395","type":"DataRange1d"},{"attributes":{},"id":"17416","type":"UndoTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"17411"},{"id":"17412"},{"id":"17413"},{"id":"17414"},{"id":"17415"},{"id":"17416"},{"id":"17417"},{"id":"17418"}]},"id":"17421","type":"Toolbar"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"17431","type":"Circle"},{"attributes":{},"id":"17452","type":"UnionRenderers"},{"attributes":{},"id":"17453","type":"Selection"},{"attributes":{"below":[{"id":"17403"}],"center":[{"id":"17406"},{"id":"17410"}],"left":[{"id":"17407"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"17433"},{"id":"17437"},{"id":"17439"},{"id":"17440"}],"title":{"id":"17441"},"toolbar":{"id":"17421"},"toolbar_location":null,"x_range":{"id":"17395"},"x_scale":{"id":"17399"},"y_range":{"id":"17397"},"y_scale":{"id":"17401"}},"id":"17394","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"17404","type":"BasicTicker"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAAACamZmZmZmpP5qZmZmZmbk/NDMzMzMzwz+amZmZmZnJPwAAAAAAANA/NDMzMzMz0z9nZmZmZmbWP5qZmZmZmdk/zczMzMzM3D8AAAAAAADgP5qZmZmZmeE/NDMzMzMz4z/NzMzMzMzkP2dmZmZmZuY/AAAAAAAA6D+amZmZmZnpPzQzMzMzM+s/zczMzMzM7D9nZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"PSWUrfrllUC6ZTfE07ObQIHSxpixD6BAXutlA2Fon0AWemawUmmdQBJdptJ4v59A48F0NdPPm0BhLg6FQP6cQNT8SMHKBJ1AN/hQuCybnECYCNGlG7+ZQOL2UqcslZtAhDgC8IHHmkDe0e+ORHOcQJp7JDyiB6BAxVCwvOY1n0BSmX0dHQ6gQFTe5JuCLJtAHVMIAw/Em0COFXr8JyGYQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"17451"},"selection_policy":{"id":"17450"}},"id":"17430","type":"ColumnDataSource"},{"attributes":{"children":[{"id":"17457"},{"id":"17455"}]},"id":"17458","type":"Column"},{"attributes":{"data_source":{"id":"17430"},"glyph":{"id":"17431"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17432"},"selection_glyph":null,"view":{"id":"17434"}},"id":"17433","type":"GlyphRenderer"},{"attributes":{"source":{"id":"17436"}},"id":"17438","type":"CDSView"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"17419","type":"BoxAnnotation"},{"attributes":{"source":{"id":"17430"}},"id":"17434","type":"CDSView"},{"attributes":{},"id":"17447","type":"BasicTickFormatter"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"17440","type":"Span"},{"attributes":{"axis_label":"ESS for small intervals","formatter":{"id":"17447"},"ticker":{"id":"17408"}},"id":"17407","type":"LinearAxis"},{"attributes":{"data_source":{"id":"17436"},"glyph":{"id":"17435"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"17438"}},"id":"17437","type":"GlyphRenderer"},{"attributes":{"data":{"rug_x":{"__ndarray__":"u/QnuP7V5z9SIsxa1SLgPw==","dtype":"float64","order":"little","shape":[2]},"rug_y":{"__ndarray__":"nB0+wbWyacCcHT7BtbJpwA==","dtype":"float64","order":"little","shape":[2]}},"selected":{"id":"17453"},"selection_policy":{"id":"17452"}},"id":"17436","type":"ColumnDataSource"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"17420","type":"PolyAnnotation"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"17439","type":"Span"},{"attributes":{},"id":"17399","type":"LinearScale"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"17445"},"ticker":{"id":"17404"}},"id":"17403","type":"LinearAxis"},{"attributes":{"toolbars":[{"id":"17421"}],"tools":[{"id":"17411"},{"id":"17412"},{"id":"17413"},{"id":"17414"},{"id":"17415"},{"id":"17416"},{"id":"17417"},{"id":"17418"}]},"id":"17456","type":"ProxyToolbar"},{"attributes":{"callback":null},"id":"17418","type":"HoverTool"}],"root_ids":["17458"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"ed527f2a-c25d-4d8d-bf65-4f65897e9b3a","root_ids":["17458"],"roots":{"17458":"05280ebc-0f00-448b-8659-14c35b70e79d"}}];
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