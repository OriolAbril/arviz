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
    
      
      
    
      var element = document.getElementById("243ed109-711d-4c39-9a4d-fb936545c415");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '243ed109-711d-4c39-9a4d-fb936545c415' but no matching script tag was found.")
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
                    
                  var docs_json = '{"fa1d6684-92e1-4891-92e7-1c9a90914b40":{"roots":{"references":[{"attributes":{"toolbars":[{"id":"4474"}],"tools":[{"id":"4464"},{"id":"4465"},{"id":"4466"},{"id":"4467"},{"id":"4468"},{"id":"4469"},{"id":"4470"},{"id":"4471"}]},"id":"4502","type":"ProxyToolbar"},{"attributes":{},"id":"4467","type":"WheelZoomTool"},{"attributes":{},"id":"4470","type":"SaveTool"},{"attributes":{},"id":"4469","type":"UndoTool"},{"attributes":{"below":[{"id":"4456"}],"center":[{"id":"4459"},{"id":"4463"}],"left":[{"id":"4460"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"4486"},{"id":"4488"}],"title":{"id":"4489"},"toolbar":{"id":"4474"},"toolbar_location":null,"x_range":{"id":"4448"},"x_scale":{"id":"4452"},"y_range":{"id":"4450"},"y_scale":{"id":"4454"}},"id":"4447","subtype":"Figure","type":"Plot"},{"attributes":{"callback":null},"id":"4471","type":"HoverTool"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"4473","type":"PolyAnnotation"},{"attributes":{},"id":"4450","type":"DataRange1d"},{"attributes":{"overlay":{"id":"4472"}},"id":"4466","type":"BoxZoomTool"},{"attributes":{"source":{"id":"4483"}},"id":"4487","type":"CDSView"},{"attributes":{"axis":{"id":"4456"},"ticker":null},"id":"4459","type":"Grid"},{"attributes":{},"id":"4448","type":"DataRange1d"},{"attributes":{},"id":"4454","type":"LinearScale"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4485","type":"Circle"},{"attributes":{},"id":"4492","type":"BasicTickFormatter"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4484","type":"Circle"},{"attributes":{},"id":"4498","type":"Selection"},{"attributes":{},"id":"4499","type":"UnionRenderers"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"4464"},{"id":"4465"},{"id":"4466"},{"id":"4467"},{"id":"4468"},{"id":"4469"},{"id":"4470"},{"id":"4471"}]},"id":"4474","type":"Toolbar"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"R2riOHxslUDJrxt4rb2XQC8UfR7VtJlAMRHUGWndmUBKO3TVSyObQNp1b/0mJp1ASsbCePPwnkB+Du/cq5qgQH8ihBoHoKBAjJ8qLZB5oECYlOwhLnyfQD8CvMP22p1A58Gm42rqnEALInuU09KdQICWYY7d25xA2A/0ZSlsnEBzLBEly1mdQE4F40OedZlAKS7heDC7m0BmB8tcKnmYQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4498"},"selection_policy":{"id":"4499"}},"id":"4483","type":"ColumnDataSource"},{"attributes":{},"id":"4457","type":"BasicTicker"},{"attributes":{"children":[[{"id":"4447"},0,0]]},"id":"4501","type":"GridBox"},{"attributes":{},"id":"4465","type":"PanTool"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"4494"},"ticker":{"id":"4457"}},"id":"4456","type":"LinearAxis"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"4472","type":"BoxAnnotation"},{"attributes":{"axis":{"id":"4460"},"dimension":1,"ticker":null},"id":"4463","type":"Grid"},{"attributes":{},"id":"4494","type":"BasicTickFormatter"},{"attributes":{},"id":"4464","type":"ResetTool"},{"attributes":{"axis_label":"ESS for quantiles","formatter":{"id":"4492"},"ticker":{"id":"4461"}},"id":"4460","type":"LinearAxis"},{"attributes":{},"id":"4452","type":"LinearScale"},{"attributes":{"data_source":{"id":"4483"},"glyph":{"id":"4484"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4485"},"selection_glyph":null,"view":{"id":"4487"}},"id":"4486","type":"GlyphRenderer"},{"attributes":{"children":[{"id":"4503"},{"id":"4501"}]},"id":"4504","type":"Column"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"4488","type":"Span"},{"attributes":{},"id":"4461","type":"BasicTicker"},{"attributes":{"overlay":{"id":"4473"}},"id":"4468","type":"LassoSelectTool"},{"attributes":{"toolbar":{"id":"4502"},"toolbar_location":"above"},"id":"4503","type":"ToolbarBox"},{"attributes":{"text":"sigma"},"id":"4489","type":"Title"}],"root_ids":["4504"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"fa1d6684-92e1-4891-92e7-1c9a90914b40","root_ids":["4504"],"roots":{"4504":"243ed109-711d-4c39-9a4d-fb936545c415"}}];
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